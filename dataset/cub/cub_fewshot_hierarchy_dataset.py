import csv
import glob
import os
import random
import re
from collections import defaultdict
from itertools import groupby, cycle
import math

import torch
from tqdm import tqdm

from dataset.dataset import Dataset, BuilderDataset
from models.programs import build_program
from utils import collate_fn, num2word, dump, join, mkdir, symlink_recursive

import copy
import random

import torch
from tqdm import tqdm

from dataset.cub.cub_dataset import CubDataset, CubBuilderDataset
from dataset.cub.cub_fewshot_dataset import CubFewshotDataset
from dataset.meta_dataset import MetaDataset, MetaBuilderDataset

from utils import file_cached, join, nonzero

class CubFewshotHierarchyDataset(CubFewshotDataset):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.concept2splits[self.question_concepts]
        self.indices_split = self.select_split(self.split_specs)

    @property
    def question_concepts(self):
        return torch.tensor([q['concept_index'][0] for q in self.questions])

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} "
            f"should already "
            f"exist.")

    def __getitem__(self, index):
        question_index = self.indices_split[index]
        question = self.questions[question_index]
        concept_index = question['concept_index']
        train_concept_index = concept_index[0]
        val_concept_index = concept_index
        train_samples = question['train_sample']
        train_stacked_scenes = collate_fn([self.get_stacked_scenes(i) for i in train_samples['image_index']])
        train_tokenized_statements = collate_fn(
            [self.tokenize_statement(t, train_concept_index) for t in train_samples['text']])
        train_program = [build_program(p) for p in train_samples['program']]
        train_image_indices = train_samples['image_index'] if self.is_fewshot else [-1] * len(
            train_samples['image_index'])
        train_samples = {**train_samples, **train_stacked_scenes, **train_tokenized_statements,
            'program': train_program, 'image_index': train_image_indices}

        val_samples = question['val_sample']
        val_stacked_scenes = collate_fn([self.get_stacked_scenes(i) for i in val_samples['image_index']])
        val_tokenized_question = collate_fn([self.tokenize_question(t, a, c) for t, a, c in
            zip(val_samples['text'], val_samples['answer'], val_concept_index)])
        val_program = [build_program(p).register_token(c) for p,c in zip(val_samples['program'], val_concept_index)]
        val_samples = {**val_samples, **val_stacked_scenes, **val_tokenized_question, 'program': val_program}

        metaconcept_tokenized, metaconcept_length = self.encode_unknown(train_samples['metaconcept_text'],
            train_concept_index)
        metaconcept_tokenized = torch.tensor(metaconcept_tokenized)
        composite_tokenized, composite_length, composite_segment = list(zip(*(
            self.encode_composite(train_samples['text'][0], train_samples['metaconcept_text'], val_text,
                train_concept_index) for val_text in val_samples["text"])))
        task = {'metaconcept_tokenized': metaconcept_tokenized, 'metaconcept_length': metaconcept_length,
            'composite_tokenized': torch.tensor(composite_tokenized), 'composite_length': composite_length,
            'composite_segment': torch.tensor(composite_segment)}

        if self.is_attached:
            program = build_program(("Composite", question['supports'], question['relations'], train_concept_index))
        else:
            program = build_program(('Composite', [], [], train_concept_index))

        return {**question, 'train_sample': train_samples, 'val_sample': val_samples, 'task': task,
            'program': program, 'index': index, 'question_index': question_index, 'info': self.info}

    def batch_meta_handler(self, inputs, outputs):
        accuracies = []
        predictions = []
        targets = []
        concept_index = inputs["concept_index"]
        queried_embedding = outputs["queried_embedding"]
        for i, categories in enumerate(inputs["val_sample"]["category"]):
            val_end = outputs["val_sample"]["end"][i]
            val_target = inputs["val_sample"]["target"][i]
            val_token = inputs["val_sample"]["answer_tokenized"][i]
            piece_accuracy = []
            for j, category in enumerate(categories):
                if category == "boolean":
                    pa = (~torch.logical_xor(val_end[j] > 0, val_target[j])).float()
                elif category == "count":
                    pa = (val_end[j] == val_target[j]).float()
                elif category == "choice":
                    pa = (val_end[j].max(-1).indices == val_target[j]).float()
                elif category == "token":
                    pa = (val_end[j].max(-1).indices == val_token[j]).float()
                else:
                    raise NotImplementedError
                piece_accuracy.append(pa)
            predictions.append((torch.cat(val_end) > 0).float())
            targets.append(val_target)
            accuracies.append(torch.stack(piece_accuracy).mean())
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        concept_index = torch.tensor(concept_index)
        out = {"accuracy": accuracies, "concept_index": concept_index, "queried_embedding": queried_embedding, "predictions": predictions, "targets": targets}
        return out

    def metric_meta_handler(self, evaluated):
        predictions = torch.stack(evaluated["predictions"])
        targets = torch.stack(evaluated["targets"])
        accuracy = torch.stack(evaluated["accuracy"])
        metrics = {}
        concepts = torch.cat(evaluated["concept_index"])
        split_idx = {0:"train", 1:"val", 2:"test", -1: "pretrain"}
        kinds = self.concept2kinds[concepts]
        unique_kinds = torch.unique(kinds).tolist()
        splits = self.concept2splits[concepts].int()
        unique_splits = [-1, 0, 1, 2]

        for unique_kind in unique_kinds:
            kind_predictions = predictions[kinds == unique_kind]
            kind_targets = targets[kinds == unique_kind]
            metrics[f"accuracy_{self.kinds_[unique_kind]}"] = (
                ~torch.logical_xor(kind_predictions, kind_targets)).float().mean()
            kind_tp = kind_predictions[kind_targets == 1].float().sum()
            kind_fp = kind_predictions[kind_targets == 0].float().sum()
            kind_fn = kind_targets[kind_predictions == 0].float().sum()
            metrics[f"precision_{self.kinds_[unique_kind]}"] = kind_tp / (kind_tp + kind_fp)
            metrics[f"recall_{self.kinds_[unique_kind]}"] = kind_tp / (kind_tp + kind_fn)
        for unique_split in unique_splits:
            split_predictions = predictions[splits == unique_split]
            split_targets = targets[splits == unique_split]
            metrics[f"accuracy_{split_idx[unique_split]}"] = (
                ~torch.logical_xor(split_predictions, split_targets)).float().mean()
            split_tp = split_predictions[split_targets == 1].float().sum()
            split_fp = split_predictions[split_targets == 0].float().sum()
            split_fn = split_targets[split_predictions == 0].float().sum()
            metrics[f"precision_{split_idx[unique_split]}"] = split_tp / (split_tp + split_fp)
            metrics[f"recall_{split_idx[unique_split]}"] = split_tp / (split_tp + split_fn)
        for k in unique_kinds:
            for s in unique_splits:
                split_kind_predictions = predictions[torch.logical_and(splits == s, kinds == k)]
                split_kind_targets = targets[torch.logical_and(splits == s, kinds == k)]
                metrics[f"accuracy_{split_idx[s]}_{self.kinds_[k]}"] = (
                    ~torch.logical_xor(split_kind_predictions, split_kind_targets)).float().mean()
                split_kind_tp = split_kind_predictions[split_kind_targets == 1].float().nansum()
                split_kind_fp = split_kind_predictions[split_kind_targets == 0].float().nansum()
                split_kind_fn = split_kind_targets[split_kind_predictions == 0].float().nansum()
                if split_kind_tp + split_kind_fp != 0:
                    metrics[f"precision_{split_idx[s]}_{self.kinds_[k]}"] = split_kind_tp / (
                                split_kind_tp + split_kind_fp)
                if split_kind_tp + split_kind_fn != 0:
                    metrics[f"recall_{split_idx[s]}_{self.kinds_[k]}"] = split_kind_tp / (split_kind_tp + split_kind_fn)
        tp = predictions[targets == 1].float().sum()
        fp = predictions[targets == 0].float().sum()
        fn = targets[predictions == 0].float().sum()
        metrics['precision'] = tp/(tp+fp)
        metrics['recall'] = tp/(tp+fn)

        metrics['accuracy_999'] = (~torch.logical_xor(predictions, targets)).float().mean()
        metrics['accuracy_006'] = metrics['accuracy_999']
        metrics['accuracy_001'] = metrics['accuracy_999']
        return metrics

    def get_batch_sampler(self, batch_size):
        if self.split == "test":
            return OnePerConceptSequentialBatchSampler(self, batch_size)
        else:
            return OnePerConceptRandomBatchSampler(self, batch_size)

class OnePerConceptRandomBatchSampler:
    def __init__(self, dataset, batch_size):
        assert batch_size == 1, "Fewshot doesn't support batch_size != 1"
        self.batch_size = batch_size
        unique_indices = [0] + dataset.question_concepts[dataset.indices_split].unique_consecutive(
            return_counts=True)[-1].cumsum(-1)[:-1].tolist()
        split_specs = dataset.concept_split_specs[
            dataset.question_concepts[list(dataset.indices_split[i] for i in unique_indices)]]
        self.segments = [[unique_indices[n] for n in nonzero(split_specs == s)] for s in
            split_specs.unique().sort().values.tolist()]
        self.length = len(dataset)

    def __iter__(self):
        for segment in self.segments:
            # randomly re-assign the order of the concepts at the same level
            concepts = copy.deepcopy(segment)
            random.shuffle(concepts)
            for i in range(0, len(concepts), self.batch_size):
                # In the dataset we built, each have 5 samples for each concept to insert
                # randomly pick one for this current concept
                index = concepts[i] + random.sample(range(5),1)[0]
                yield [index]

    def __len__(self):
        return sum(math.ceil(len(segment) / self.batch_size) for segment in self.segments)

class OnePerConceptSequentialBatchSampler:
    def __init__(self, dataset, batch_size):
        assert batch_size == 1, "Fewshot doesn't support batch_size != 1"
        self.batch_size = batch_size
        unique_indices = [0] + dataset.question_concepts[dataset.indices_split].unique_consecutive(
            return_counts=True)[-1].cumsum(-1)[:-1].tolist()
        split_specs = dataset.concept_split_specs[
            dataset.question_concepts[list(dataset.indices_split[i] for i in unique_indices)]]
        self.segments = [[unique_indices[n] for n in nonzero(split_specs == s)] for s in
            split_specs.unique().sort().values.tolist()]
        self.length = len(dataset)

    def __iter__(self):
        for j in range(5):
            for segment in self.segments:
                # randomly re-assign the order of the concepts at the same level
                concepts = copy.deepcopy(segment)
                for i in range(0, len(concepts), self.batch_size):
                    # In the dataset we built, each have 5 samples for each concept to insert
                    # randomly pick one for this current concept
                    index = concepts[i] + j
                    yield [index]

    def __len__(self):
        return self.length