import copy
from itertools import groupby
import random

import torch
from tqdm import tqdm

from dataset.tabletop.tabletop_dataset import TabletopDataset, TabletopBuilderDataset
from dataset.pretrain_dataset import PretrainBuilderDataset, PretrainDataset
from dataset.utils import FixedResizeTransform, sample_with_ratio
from models.programs import build_program, to_batch
from utils import file_cached, join, nonzero, dump, mkdir

class TabletopConceptDataset(TabletopDataset, PretrainDataset):
    @property
    def transform_fn(self):
        return FixedResizeTransform

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

    def __getitem__(self, index):
        question_index = self.indices_split[index]
        question = self.questions[question_index]
        program = build_program(question['program'])
        stacked_scenes = self.get_stacked_scenes(question['image_index'])
        tokenized_support = self.tokenize_support([question['text']], [question['answer']])
        return {**tokenized_support, **stacked_scenes, **question, 'program': program, 'index': index,
            'question_index': question_index, 'info': self.info}

        #return { **stacked_scenes, **question, 'program': program, 'index': index,
        #        'question_index': question_index, 'info': self.info}
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
        questioned_concepts = inputs['questioned_concept']
        target = inputs['answer']
        predictions = outputs['end']
        return {"accuracy": accuracies, "questioned_concepts": questioned_concepts, "targets": target, "predictions": predictions}


    def metric_pretrain_handler(self, evaluated):

        accuracy = torch.stack(evaluated["accuracy"]).mean()
        metrics = {"accuracy": accuracy}
        predictions = (torch.stack(evaluated["predictions"]) > 0).squeeze(1)

        targets = torch.tensor(evaluated["targets"])
        concepts = torch.tensor(evaluated['questioned_concepts'])
        splits = self.concept2splits[concepts].int()
        unique_splits = [-1, 0, 1]
        split_idx = {0: "train", 1: "val", 2: "test", -1: "pretrain"}
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
        if self.domain == "house":
            color_predictions = predictions[torch.logical_and((concepts < 40),(concepts >= 34))]
            color_targets = targets[torch.logical_and((concepts < 40),(concepts >= 34))]
            affordance_predictions = predictions[torch.logical_and((concepts < 34),(concepts >= 31))]
            affordance_targets = targets[torch.logical_and((concepts < 34),(concepts >= 31))]
            object_predictions = predictions[concepts < 31]
            object_targets = targets[concepts < 31]
            object_tp = object_predictions[object_targets == 1].float().sum()
            object_fp = object_predictions[object_targets == 0].float().sum()
            object_fn = object_targets[object_predictions == 0].float().sum()
            color_tp = color_predictions[color_targets == 1].float().sum()
            color_fp = color_predictions[color_targets == 0].float().sum()
            color_fn = color_targets[color_predictions == 0].float().sum()
            affordance_tp = affordance_predictions[affordance_targets == 1].float().sum()
            affordance_fp = affordance_predictions[affordance_targets == 0].float().sum()
            affordance_fn = affordance_targets[affordance_predictions == 0].float().sum()
            metrics['accuracy_object'] = (~torch.logical_xor(object_predictions, object_targets)).float().mean()
            metrics[f"precision_object"] = object_tp / (object_tp + object_fp)
            metrics[f"recall_object"] = object_tp / (object_tp + object_fn)
            metrics['accuracy_color'] = (~torch.logical_xor(color_predictions, color_targets)).float().mean()
            metrics[f"precision_color"] = color_tp / (color_tp + color_fp)
            metrics[f"recall_color"] = color_tp / (color_tp + color_fn)
            metrics['accuracy_affordance'] = (~torch.logical_xor(affordance_predictions, affordance_targets)).float().mean()
            metrics[f"precision_affordance"] = affordance_tp / (affordance_tp + affordance_fp)
            metrics[f"recall_affordance"] = affordance_tp / (affordance_tp + affordance_fn)
        tp = predictions[targets == 1].float().sum()
        fp = predictions[targets == 0].float().sum()
        fn = targets[predictions == 0].float().sum()
        metrics['precision'] = tp / (tp + fp)
        metrics['recall'] = tp / (tp + fn)
        return metrics


class TabletopConceptSupportDataset(TabletopConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

class TabletopConceptWarmupSupportDataset(TabletopConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")


class TabletopHumanSubjectStudyConceptSupportDataset(TabletopConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

class TabletopConceptFullDataset(TabletopConceptDataset):

    @file_cached('questions')
    def _build_questions(self):
        raise FileNotFoundError(
            f"{join(self.augmented_root, self.get_augmented_name(__class__.__qualname__), 'questions')} should already "
            f"exist.")

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
        questioned_concepts = inputs['questioned_concept']
        target = inputs['answer']
        predictions = outputs['end']
        return {"accuracy": accuracies, "questioned_concepts": questioned_concepts, "targets": target, "predictions": predictions}


    def metric_pretrain_handler(self, evaluated):

        accuracy = torch.stack(evaluated["accuracy"]).mean()
        metrics = {"accuracy": accuracy}
        predictions = (torch.stack(evaluated["predictions"]) > 0).squeeze(1)

        targets = torch.tensor(evaluated["targets"])
        concepts = torch.tensor(evaluated['questioned_concepts'])
        splits = self.concept2splits[concepts].int()
        unique_splits = [-1, 0, 1]
        split_idx = {0: "train", 1: "val", 2: "test", -1: "pretrain"}
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
        if self.domain == "house":
            color_predictions = predictions[torch.logical_and((concepts < 40),(concepts >= 34))]
            color_targets = targets[torch.logical_and((concepts < 40),(concepts >= 34))]
            affordance_predictions = predictions[torch.logical_and((concepts < 34),(concepts >= 31))]
            affordance_targets = targets[torch.logical_and((concepts < 34),(concepts >= 31))]
            object_predictions = predictions[concepts < 31]
            object_targets = targets[concepts < 31]
            object_tp = object_predictions[object_targets == 1].float().sum()
            object_fp = object_predictions[object_targets == 0].float().sum()
            object_fn = object_targets[object_predictions == 0].float().sum()
            color_tp = color_predictions[color_targets == 1].float().sum()
            color_fp = color_predictions[color_targets == 0].float().sum()
            color_fn = color_targets[color_predictions == 0].float().sum()
            affordance_tp = affordance_predictions[affordance_targets == 1].float().sum()
            affordance_fp = affordance_predictions[affordance_targets == 0].float().sum()
            affordance_fn = affordance_targets[affordance_predictions == 0].float().sum()
            metrics['accuracy_object'] = (~torch.logical_xor(object_predictions, object_targets)).float().mean()
            metrics[f"precision_object"] = object_tp / (object_tp + object_fp)
            metrics[f"recall_object"] = object_tp / (object_tp + object_fn)
            metrics['accuracy_color'] = (~torch.logical_xor(color_predictions, color_targets)).float().mean()
            metrics[f"precision_color"] = color_tp / (color_tp + color_fp)
            metrics[f"recall_color"] = color_tp / (color_tp + color_fn)
            metrics['accuracy_affordance'] = (~torch.logical_xor(affordance_predictions, affordance_targets)).float().mean()
            metrics[f"precision_affordance"] = affordance_tp / (affordance_tp + affordance_fp)
            metrics[f"recall_affordance"] = affordance_tp / (affordance_tp + affordance_fn)
        if self.domain == "park":
            leaf_predictions = predictions[concepts <= 27]
            leaf_targets = targets[concepts <= 27]
            non_leaf_predictions = predictions[concepts > 27]
            non_leaf_targets = targets[concepts > 27]
            leaf_tp = leaf_predictions[leaf_targets == 1].float().sum()
            leaf_fp = leaf_predictions[leaf_targets == 0].float().sum()
            leaf_fn = leaf_targets[leaf_predictions == 0].float().sum()
            non_leaf_tp = non_leaf_predictions[non_leaf_targets == 1].float().sum()
            non_leaf_fp = non_leaf_predictions[non_leaf_targets == 0].float().sum()
            non_leaf_fn = non_leaf_targets[non_leaf_predictions == 0].float().sum()
            metrics['accuracy_leaf'] = (~torch.logical_xor(leaf_predictions, leaf_targets)).float().mean()
            metrics[f"precision_leaf"] = leaf_tp / (leaf_tp + leaf_fp)
            metrics[f"recall_leaf"] = leaf_tp / (leaf_tp + leaf_fn)
            metrics['accuracy_non_leaf'] = (~torch.logical_xor(non_leaf_predictions, non_leaf_targets)).float().mean()
            metrics[f"precision_non_leaf"] = non_leaf_tp / (non_leaf_tp + non_leaf_fp)
            metrics[f"recall_non_leaf"] = non_leaf_tp / (non_leaf_tp + non_leaf_fn)
        if self.domain == "house_study":
            color_predictions = predictions[torch.logical_and((concepts < 22),(concepts >= 16))]
            color_targets = targets[torch.logical_and((concepts < 22),(concepts >= 16))]
            affordance_predictions = predictions[torch.logical_and((concepts < 25),(concepts >= 22))]
            affordance_targets = targets[torch.logical_and((concepts < 25),(concepts >= 22))]
            object_predictions = predictions[concepts < 16]
            object_targets = targets[concepts < 16]
            object_tp = object_predictions[object_targets == 1].float().sum()
            object_fp = object_predictions[object_targets == 0].float().sum()
            object_fn = object_targets[object_predictions == 0].float().sum()
            color_tp = color_predictions[color_targets == 1].float().sum()
            color_fp = color_predictions[color_targets == 0].float().sum()
            color_fn = color_targets[color_predictions == 0].float().sum()
            affordance_tp = affordance_predictions[affordance_targets == 1].float().sum()
            affordance_fp = affordance_predictions[affordance_targets == 0].float().sum()
            affordance_fn = affordance_targets[affordance_predictions == 0].float().sum()
            metrics['accuracy_object'] = (~torch.logical_xor(object_predictions, object_targets)).float().mean()
            metrics[f"precision_object"] = object_tp / (object_tp + object_fp)
            metrics[f"recall_object"] = object_tp / (object_tp + object_fn)
            metrics['accuracy_color'] = (~torch.logical_xor(color_predictions, color_targets)).float().mean()
            metrics[f"precision_color"] = color_tp / (color_tp + color_fp)
            metrics[f"recall_color"] = color_tp / (color_tp + color_fn)
            metrics['accuracy_affordance'] = (~torch.logical_xor(affordance_predictions, affordance_targets)).float().mean()
            metrics[f"precision_affordance"] = affordance_tp / (affordance_tp + affordance_fp)
            metrics[f"recall_affordance"] = affordance_tp / (affordance_tp + affordance_fn)
        tp = predictions[targets == 1].float().sum()
        fp = predictions[targets == 0].float().sum()
        fn = targets[predictions == 0].float().sum()
        metrics['precision'] = tp / (tp + fp)
        metrics['recall'] = tp / (tp + fn)
        return metrics

class TabletopPretrainBuilderDataset(PretrainBuilderDataset, TabletopBuilderDataset):
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

    def _build_image_questions(self, object, image_index):
        image_annotation = self.image_annotations[image_index]
        #if len(image_annotation["objects"]) == 1:
            # ask boolean question only
        questions = []
        candidates = [self.entry2idx_[r['related_concept']] for r in self.relations[self.concepts_[object]] if
                      r['relation_type'] != 'has_attribute'] + [object.item()]

        positive_concepts = random.choices(candidates, k=self.N_SAMPLES)
        negative_concepts = random.choices(list(set(self.concept_sets).difference(candidates)),
                                           k=self.N_SAMPLES)
        answers = [True] * len(positive_concepts) + [False] * len(negative_concepts)

        for concept, a in zip(positive_concepts + negative_concepts, answers):
            question = self.exist_question([concept])
            question_encoded, question_length = self.encode_text(question)
            question_program = self.exist_question_program([concept])
            question_target, _ = self.encode_program(question_program)
            questions.append({"question": question, "answer": a, "image_index": image_index,
                              "question_target": question_target, "question_encoded": question_encoded,
                              'question_length': question_length, "questioned_concept": concept,
                              "question_program": question_program})
            # here we are just trying to generate some question to train the root attribute concepts
        #     answers = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
        #     concept_1s = random.choices([self.entry2idx_[x] for x in self.attributes['affordance']], k=10) + \
        #                  random.choices([self.entry2idx_[x] for x in self.attributes['color']], k=10)
        #
        #     attribute_concepts = [self.entry2idx_['affordance']] * 5 + [self.entry2idx_['color']] * 5 + [
        #         self.entry2idx_['affordance']] * 5 + [self.entry2idx_['color']] * 5
        #
        #     for c1, c2, a in zip(concept_1s, attribute_concepts, answers):
        #         if 'full' in self.name:
        #             break
        #         question = self.is_instance_of_question(c1, c2)
        #         question_encoded, question_length = self.encode_text(question)
        #         question_program = self.is_instance_of_question_program(c1, c2)
        #         question_target, _ = self.encode_program(question_program)
        #         questions.append({"question": question, "answer": a, "image_index": image_index,
        #                           "question_target": question_target, "question_encoded": question_encoded,
        #                           'question_length': question_length, "questioned_concept": concept,
        #                           "question_program": question_program})
        #
        # else:
        #     # right now we just skip it
        #     # TODO: @Weiwei add some questions
        #     questions = []

        return questions

    # def _build_image_questions(self, object, image_index):
    #     image_annotation = self.image_annotations[image_index]
    #     object_color = \
    #     [r["related_concept"] for r in self.relations[self.concepts_[object]] if r['relation_type'] == 'has_color'][0]
    #     object_afford = [r["related_concept"] for r in self.relations[self.concepts_[object]] if
    #                      r['relation_type'] == 'has_affordance'][0]
    #     negative_concept_candidate_of_same_color = []
    #     negative_concept_candidate_of_same_affordance = []
    #     for obj in self.objects_:
    #         if (obj == self.concepts_[object]) or (self.entry2idx_[obj] not in self.concept_sets):
    #             continue
    #         relations = self.relations[obj]
    #         color = [r['related_concept'] for r in relations if r['relation_type'] == 'has_color'][0]
    #         affordance = [r['related_concept'] for r in relations if r['relation_type'] == 'has_affordance'][0]
    #         if color == object_color:
    #             negative_concept_candidate_of_same_color.append(self.entry2idx_[obj])
    #         elif affordance == object_afford:
    #             negative_concept_candidate_of_same_affordance.append(self.entry2idx_[obj])
    #         else:
    #             continue
    #     if len(image_annotation["objects"]) == 1:
    #         # ask boolean question only
    #         questions = []
    #         candidates = [self.entry2idx_[r['related_concept']] for r in self.relations[self.concepts_[object]] if r['relation_type'] != 'has_attribute'] + [object.item()]
    #         positive_concepts = random.choices(candidates, 1 / self.concept_frequencies[candidates],
    #                                         k=self.N_SAMPLES)
    #         if (len(negative_concept_candidate_of_same_affordance) == 0) or (len(negative_concept_candidate_of_same_color) == 0):
    #             negative_concepts = random.choices(list(set(self.concept_sets).difference(candidates)),
    #                                         k=self.N_SAMPLES)
    #         else:
    #             negative_concepts = []
    #             negative_concepts.extend(random.choices(negative_concept_candidate_of_same_color, k=self.N_SAMPLES//4))
    #             negative_concepts.extend(random.choices(negative_concept_candidate_of_same_affordance, k=self.N_SAMPLES//4))
    #             negative_concepts.extend(random.choices(list(set(self.concept_sets).difference(candidates)),
    #                                                 k=self.N_SAMPLES//2))
    #         answers = [True] * len(positive_concepts) + [False] * len(negative_concepts)
    #
    #         for concept, a in zip(positive_concepts + negative_concepts, answers):
    #             question = self.exist_question([concept])
    #             question_encoded, question_length = self.encode_text(question)
    #             question_program = self.exist_question_program([concept])
    #             question_target, _ = self.encode_program(question_program)
    #             questions.append({"question": question, "answer": a, "image_index": image_index,
    #                               "question_target": question_target, "question_encoded": question_encoded,
    #                               'question_length': question_length, "questioned_concept": concept, "question_program": question_program})
    #         # here we are just trying to generate some question to train the root attribute concepts
    #         answers = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
    #         concept_1s = random.choices([self.entry2idx_[x] for x in self.attributes['affordance']], k=10) + \
    #                         random.choices([self.entry2idx_[x] for x in self.attributes['color']], k=10)
    #
    #         attribute_concepts = [self.entry2idx_['affordance']] * 5 + [self.entry2idx_['color']] * 5 + [self.entry2idx_['affordance']] * 5 + [self.entry2idx_['color']] * 5
    #
    #         for c1, c2, a in zip(concept_1s, attribute_concepts, answers):
    #             if ('full' in self.name) or (self.split_seed == 999):
    #                 break
    #             question = self.is_instance_of_question(c1, c2)
    #             question_encoded, question_length = self.encode_text(question)
    #             question_program = self.is_instance_of_question_program(c1, c2)
    #             question_target, _ = self.encode_program(question_program)
    #             questions.append({"question": question, "answer": a, "image_index": image_index,
    #                               "question_target": question_target, "question_encoded": question_encoded,
    #                               'question_length': question_length, "questioned_concept": c2, "question_program": question_program})
    #
    #     else:
    #         # right now we just skip it
    #         # TODO: @Weiwei add some questions
    #         questions = []
    #
    #     return questions

    def dump_questions(self):
        output_dir = f'{self.augmented_root}/{self.name[:-8]}'
        mkdir(output_dir)
        questions = []
        for q in self.questions:
            program =q['question_program']
            program = self.nscl2program(self.clevr2nscl(program))
            resulted_q = {
                "text": q['question'],
                'program': program,
                'answer': q['answer'],
                'image_index': q['image_index'],
                'length': q['question_length'],
                'questioned_concept': q['questioned_concept']
            }
            questions.append(resulted_q)
        dump(questions, join(output_dir, 'questions.json'))




class TabletopConceptSupportBuilderDataset(TabletopPretrainBuilderDataset):

    @property
    def concept_sets(self):
        # pretrain: -1
        # train: 0
        # val: 1
        # test: 2
        return nonzero(self.concept2splits == -1)

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for object in self.objects:
            if self.concept2splits[object] != -1:
                continue
            obj_name = self.concepts_[object]
            related_objects = [self.entry2idx_[r['related_concept']] for r in self.relations[obj_name] if r['relation_type'] != 'has_attribute']
            concept_frequencies[related_objects + [object]] += 1
        return concept_frequencies

    #@file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2object)):
            if self.concept2splits[cls] != -1: continue
            questions.extend(self._build_image_questions(cls, image_index))
        return questions

class TabletopConceptWarmupSupportBuilderDataset(TabletopPretrainBuilderDataset):

    @property
    def concept_sets(self):
        # pretrain: -1
        # train: 0
        # val: 1
        # test: 2
        return nonzero(self.concept2splits < 1)

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for object in self.objects:
            if self.concept2splits[object] >= 1:
                continue
            obj_name = self.concepts_[object]
            related_objects = [self.entry2idx_[r['related_concept']] for r in self.relations[obj_name] if r['relation_type'] != 'has_attribute']
            concept_frequencies[related_objects + [object]] += 1
        return concept_frequencies

    #@file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2object)):
            if self.concept2splits[cls] > 0: continue
            questions.extend(self._build_image_questions(cls, image_index))
        return questions


class TabletopConceptFullBuilderDataset(TabletopPretrainBuilderDataset):

    @property
    def concept_sets(self):
        # pretrain: -1
        # train: 0
        # val: 1
        # test: 2
        return self.concepts

    def _build_concept_frequencies(self):
        concept_frequencies = torch.zeros_like(self.concept_split_specs)
        for object in self.objects:
            obj_name = self.concepts_[object]
            related_objects = [self.entry2idx_[r['related_concept']] for r in self.relations[obj_name] if r['relation_type'] != 'has_attribute']
            concept_frequencies[related_objects + [object]] += 1
        return concept_frequencies

    #@file_cached("questions")
    def _build_questions(self):
        questions = []
        for image_index, cls in enumerate(tqdm(self.image2object)):
            questions.extend(self._build_image_questions(cls, image_index))
        return questions
