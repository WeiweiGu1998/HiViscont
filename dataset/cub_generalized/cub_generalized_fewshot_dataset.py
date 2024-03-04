import os
import random
from itertools import chain, cycle

import math
import torch
from tqdm import tqdm

from dataset.cub_generalized.cub_generalized_dataset import CubGeneralizedDataset, CubGeneralizedBuilderDataset
from dataset.meta_dataset import MetaDataset, MetaBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, join, nonzero, mkdir, dump


class CubGeneralizedFewshotDataset(CubGeneralizedDataset, MetaDataset):

    @property
    def question_concepts(self):
        return torch.tensor([q['concept_index'] for q in self.questions])

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.concept2splits[self.question_concepts]
        self.indices_split = self.select_split(self.split_specs)

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} "
            f"should already "
            f"exist.")

    def get_batch_sampler(self, batch_size):
        if self.split == "train":
            return None
        elif self.split == "val":
            return OnePerConceptBatchSampler(self, batch_size)
        elif self.split == "test":
            return ManyPerConceptBatchSampler(self, batch_size)

    def save_meta_handler(self, output_dir, evaluated, iteration, metrics):
        super().save_meta_handler(output_dir, evaluated, iteration, metrics)
        if self.split == "test" and not self.use_text:
            head, tail = os.path.split(output_dir)
            parts = tail.split("_")
            output_dir = mkdir(join(head, '_'.join(parts[:-1] + ['shallow'] + parts[-1:])))
            filename_prefix = evaluated["mode"]
            metrics = {**metrics, 'principal': metrics['accuracy_006']}
            dump(metrics, join(output_dir, f"{filename_prefix}_{iteration:07d}.json"))


class CubGeneralizedDetachedDataset(CubGeneralizedFewshotDataset):
    pass

class CubGeneralizedZeroshotDataset(CubGeneralizedFewshotDataset):
    pass

class OnePerConceptBatchSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        unique_indices = [0] + dataset.question_concepts[dataset.indices_split].unique_consecutive(
            return_counts=True)[-1].cumsum(-1)[:-1].tolist()
        split_specs = dataset.concept_split_specs[
            dataset.question_concepts[list(dataset.indices_split[i] for i in unique_indices)]]
        self.segments = [[unique_indices[n] for n in nonzero(split_specs == s)] for s in
            split_specs.unique().sort().values.tolist()]

    def __iter__(self):
        for segment in self.segments:
            for i in range(0, len(segment), self.batch_size):
                yield segment[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(segment) / self.batch_size) for segment in self.segments)


class ManyPerConceptBatchSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        unique_indices = torch.arange(len(dataset)).tolist()
        split_specs = dataset.concept_split_specs[
            dataset.question_concepts[list(dataset.indices_split[i] for i in unique_indices)]]
        self.segments = [[unique_indices[n] for n in nonzero(split_specs == s)] for s in
            split_specs.unique().sort().values.tolist()]

    def __iter__(self):
        for segment in self.segments:
            for i in range(0, len(segment), self.batch_size):
                yield segment[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(segment) / self.batch_size) for segment in self.segments)


class CubFewshotGeneralizedBuilderDataset(CubGeneralizedBuilderDataset, MetaBuilderDataset):
    N_SAMPLES = 5

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self._build_mac()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)


    #@file_cached('questions')
    def _build_questions(self):
        # Make the questions fewshot because zero-shot is handled in the model
        questions = []
        for c in tqdm(self.concepts[200:365]):
            # We only care about concepts of higher level in the hierarchy
            # Now assume that we have a hierarchy of level 2
            # This design can be easily changed
            valid_classes = set(self.concepts[:len(self.classes)])
            subclasses = sorted(valid_classes.intersection([c, *self.hyper2hypo[c]]))
            # In this method, the val sample can be from seen concept(support classes) or unseen concept(unseen classes)
            # TODO (@Weiwei) use support_candidates so that all the val samples are unseen
            true_candidates = list(chain.from_iterable(self.class2images[x] for x in subclasses))
            false_classes = list(sorted(valid_classes.difference([c, *self.hyper2hypo[c]])))
            false_candidates = list(chain.from_iterable(self.class2images[x] for x in false_classes))
            for i in range(self.N_SAMPLES):
                # pick a portion of subclasses to construct the instance
                # Here we assume that all supports are from subclasses
                num_supports = min(math.ceil(0.8 * len(subclasses)), 10)
                supports = subclasses[:num_supports]
                relations = [1] * len(supports)
                encoded_metaconcept = self.encode(self.metaconcept_text(supports, relations, c),
                            self.metaconcept_program(supports, relations), 'metaconcept')

                encoded_statement = self.encode(self.exist_statement(c),
                                                self.exist_statement_program(c), 'statement')
                encoded_question = self.encode(self.exist_question(c), self.exist_question_program(c),
                            'question')
                # Build for zero-shot for now
                true_image_index = random.choices(true_candidates, k=self.query_k // 2 + self.shot_k)
                train_image_index, true_image_index = true_image_index[:self.shot_k], true_image_index[
                                                                                      self.shot_k:]
                false_image_index = random.choices(false_candidates, k=self.query_k // 2)
                val_image_index = true_image_index + false_image_index
                answers = [True] * len(true_image_index) + [False] * len(false_image_index)
                for j, (ti, vi, answer) in enumerate(zip(cycle(train_image_index), val_image_index, answers)):
                    questions.append(
                        {**encoded_statement, **encoded_metaconcept, **encoded_question, 'answer': answer,
                         'concept_index': c, 'train_image_index': ti, 'image_index': vi,
                         'family': (c, i, j)})
        return questions

    #@file_cached('mac')
    def _build_mac(self):
        super()._build_mac()

    def mac_split(self, concept_index):
        if self.concept_split_specs[concept_index] == 0:
            return 'train'
        elif self.concept_split_specs[concept_index] == 1:
            return 'val'
        elif self.concept_split_specs[concept_index] == 2:
            return 'test'
        else:
            return None

    def dump_questions(self):
        outputs = {
            "statement_predicted": [torch.tensor(q['statement_target']) for q in self.questions],
            "metaconcept_predicted": [torch.tensor(q['metaconcept_target']) for q in self.questions],
            "question_predicted": [torch.tensor(q['question_target']) for q in self.questions]
        }
        evaluated = self.batch_inference_handler(self.questions, outputs)
        self.save_inference_handler(evaluated)

