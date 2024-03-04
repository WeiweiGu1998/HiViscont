import copy
import random

import torch
from tqdm import tqdm

from dataset.cub.cub_dataset import CubDataset, CubBuilderDataset
from dataset.pretrain_dataset import PretrainBuilderDataset, PretrainDataset
from dataset.utils import FixedCropTransform, RandomCropTransform, sample_with_ratio
from models.programs import build_program
from utils import file_cached, join, nonzero


class CubConceptDataset(CubDataset, PretrainDataset):

    @property
    def transform_fn(self):
        return RandomCropTransform if self.split == "train" else FixedCropTransform

    @property
    def question_images(self):
        return torch.tensor([q['image_index'] for q in self.questions])

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = self.image_split_specs[self.question_images]
        self.indices_split = self.select_split(self.split_specs)

    def _build_questions(self):
        raise NotImplementedError

    def batch_pretrain_handler(self, inputs, outputs):
        accuracies = []
        for i, category in enumerate(inputs["category"]):
            end = outputs["end"][i]
            target = inputs["target"][i]
            if category == "boolean":
                accuracy = (~torch.logical_xor(end > 0, target)).float().mean()
            elif category == "count":
                accuracy = (end == target).float().mean()
            else:
                accuracy = (end.max(-1).indices == target).float().mean()
            accuracies.append(accuracy)
        questioned_concepts = inputs['questioned_concept'][0]
        target = inputs['answer'][0]
        predictions = outputs['end'][0]
        return {"accuracy": accuracies, "questioned_concepts": questioned_concepts, "targets": target, "predictions": predictions}

    def metric_pretrain_handler(self, evaluated):
        accuracy = torch.stack(evaluated["accuracy"]).mean()
        metrics = {"accuracy": accuracy}
        predictions = torch.stack(evaluated["predictions"]) > 0
        targets = torch.tensor(evaluated["targets"])
        concepts = torch.tensor(evaluated['questioned_concepts'])
        kinds = self.concept2kinds[concepts]
        unique_kinds = torch.unique(kinds).tolist()
        splits = self.concept2splits[concepts].int()
        unique_splits = [-1, 0, 1]
        split_idx = {0: "train", 1: "val", 2: "test", -1: "pretrain"}
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
        tp = predictions[targets == 1].float().sum()
        fp = predictions[targets == 0].float().sum()
        fn = targets[predictions == 0].float().sum()
        metrics['precision'] = tp / (tp + fp)
        metrics['recall'] = tp / (tp + fn)
        return metrics

class CubConceptSupportDataset(CubConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class CubConceptWarmupSupportDataset(CubConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class CubConceptSupportLevelOneDataset(CubConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

class CubConceptSupportLevelTwoDataset(CubConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

class CubConceptSupportLevelThreeDataset(CubConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

class CubConceptSupportLevelFourDataset(CubConceptDataset):
    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")
class CubConceptFullDataset(CubConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

class CubConceptSpecificDataset(CubConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

class CubPretrainBuilderDataset(PretrainBuilderDataset, CubBuilderDataset):
    N_SAMPLES = 100

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.concept_frequencies = self._build_concept_frequencies()
        self.questions = self._build_questions()
        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def _build_questions(self):
        raise NotImplementedError

    @property
    def concept_sets(self):
        raise NotImplementedError

    def _build_image_questions(self, objects, image_index):
        questions = []
        candidates = objects[0]
        positive_concepts = random.choices(candidates, 1 / self.concept_frequencies[candidates],
            k=self.N_SAMPLES)
        negative_concepts = random.choices(list(set(self.concept_sets).difference(candidates)),
            k=self.N_SAMPLES)
        answers = [True] * len(positive_concepts) + [False] * len(negative_concepts)
        for concept, a in zip(positive_concepts + negative_concepts, answers):
            question = self.exist_question(concept)
            question_encoded, question_length = self.encode_text(question)
            question_program = self.exist_question_program(concept)
            question_target, _ = self.encode_program(question_program)
            questions.append({"question": question, "answer": a, "image_index": image_index,
                "question_target": question_target, "question_encoded": question_encoded,
                'question_length': question_length, "questioned_concept": concept})
        return questions


class CubConceptSupportBuilderDataset(CubPretrainBuilderDataset):

    @property
    def concept_sets(self):
        # pretrain: -1
        # train: 0
        # val: 1
        # test: 2
        return nonzero(self.concept2splits == -1)

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for hypo in nonzero((self.concept2splits == -1) & (self.concept2kinds == 0)):
            concept_frequencies[self.hypo2hyper[hypo] + [hypo]] += 1
        return concept_frequencies

    #@file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2classes)):
            if self.concept2splits[cls] != -1: continue
            questions.extend(self._build_image_questions(self.obj2concepts[image_index], image_index))
        return questions


class CubConceptWarmupSupportBuilderDataset(CubPretrainBuilderDataset):

    @property
    def concept_sets(self):
        # pretrain: -1
        # train: 0
        # val: 1
        # test: 2
        return nonzero(self.concept2splits < 1)

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for hypo in nonzero((self.concept2splits < 1) & (self.concept2kinds == 0)):
            concept_frequencies[self.hypo2hyper[hypo] + [hypo]] += 1
        return concept_frequencies

    #@file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2classes)):
            if self.concept2splits[cls] >= 1: continue
            questions.extend(self._build_image_questions(self.obj2concepts[image_index], image_index))
        return questions


class CubConceptFullBuilderDataset(CubPretrainBuilderDataset):

    @property
    def concept_sets(self):
        return self.concepts

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for hypo in nonzero(self.concept2kinds == 0):
            concept_frequencies[self.hypo2hyper[hypo] + [hypo]] += 1
        return concept_frequencies

    #@file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2classes)):
            questions.extend(self._build_image_questions(self.obj2concepts[image_index], image_index))
        return questions


class CubConceptSpecificBuilderDataset(CubPretrainBuilderDataset):

    @property
    def concept_sets(self):
        return self.concepts

    #@file_cached("questions")
    def _build_image_questions(self, objects, image_index):
        # This method should build all images vs. all concepts
        questions = []
        candidates = objects[0]
        positive_concepts = []
        # build negative samples as well
        # by choosing concept from species, genera, families, and orders
        negative_candidates = list(set(self.concept_sets).difference(candidates))
        answers = []
        for c in candidates:
            positive_concepts.extend([c] * (self.N_SAMPLES // len(candidates)))
            answers.extend([True] * (self.N_SAMPLES // len(candidates)))
        negative_concepts = random.choices(negative_candidates, k=self.N_SAMPLES)
        answers.extend([False for _ in range(len(negative_concepts))])
        concepts = positive_concepts + negative_concepts
        for concept, a in zip(concepts, answers):
            question = self.exist_question(concept)
            question_encoded, question_length = self.encode_text(question)
            question_program = self.exist_question_program(concept)
            question_target, _ = self.encode_program(question_program)
            questions.append({"question": question, "answer": a, "image_index": image_index,
                "question_target": question_target, "question_encoded": question_encoded,
                'question_length': question_length, "questioned_concept": concept})
        return questions
    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for hypo in nonzero(self.concept2kinds == 0):
            concept_frequencies[self.hypo2hyper[hypo] + [hypo]] += 1
        return concept_frequencies
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2classes)):
            # Build images
            #if cls != 155: continue
            # skip the train concepts
            if (self.concept2splits[cls] != 1) and (self.concept2splits[cls] != 2): continue
            questions.extend(self._build_image_questions(self.obj2concepts[image_index], image_index))
            #questions.extend(self._build_image_questions([[cls]], image_index))
        return questions
