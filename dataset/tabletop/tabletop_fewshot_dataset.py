import os
import random
import copy
from itertools import chain, cycle, groupby, cycle
import math
import torch
from models.programs import build_program
from tqdm import tqdm

from dataset.tabletop.tabletop_dataset import TabletopDataset, TabletopBuilderDataset
from dataset.meta_dataset import MetaDataset, MetaBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, join, nonzero, mkdir, dump,  to_serializable
from utils import collate_fn, num2word, dump, join, mkdir, symlink_recursive


class TabletopFewshotDataset(TabletopDataset, MetaDataset):

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
        # TODO: Method bound to change for this dataset
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
        splits = self.concept2splits[concepts].int()
        unique_splits = [-1, 0, 1, 2]
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
        # todo: maybe change the split specs based on the hierarchy as well
        # similar to what I did for cub
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

class TabletopFewshotBuilderDataset(TabletopBuilderDataset, MetaBuilderDataset):
    N_SAMPLES = 5

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def _build_questions(self):
        # This current versions care only about the same branch of the tree
        # We may want to include various types of edges in the future
        questions = []
        used_concepts = self.concepts
        # currently hard-code not to consider inserting new attribute
        # if "house" == self.domain:
        #     used_concepts = self.concepts[:-2]
        if self.domain == "house_study":
            num_for_oneshot = 100
        else:
            num_for_oneshot = 20

        for itself in tqdm(used_concepts):
            # we don't want to propogate "has_attribute" relations
            # mainly we don't want to adjust the embedding of abstract concepts such as "affordance" and "color" when we are inserting an object
            related_concepts = [self.entry2idx_[r['related_concept']] for r in self.relations[self.concepts_[itself]] if r['relation_type'] != 'has_attribute']
            relation_type = [r['relation_type'] for r in self.relations[self.concepts_[itself]] if r['relation_type'] != 'has_attribute']
            # check the type of current concept by checking whether is_instance_of relation in relation type
            # itself_type = "object"
            # for r in self.relations[self.concepts_[itself]]:
            #     if r['relation_type'] == 'is_instance_of':
            #         itself_type = r['related_concept']
            #         break
            #valid_object_concepts = set(nonzero(torch.logical_or((self.concept2splits[:len(self.objects)] < self.concept2splits[itself]), (self.concept2splits[:len(self.objects)] == -1))))
            #if itself_type == "object":
            #    valid_object_concepts=valid_object_concepts.union([itself])
            valid_object_concepts = set(nonzero(self.concept2splits[:len(self.objects)] <= self.concept2splits[itself]))
            #valid_object_concepts = set(nonzero(self.concept2splits[:len(self.objects)] == self.concept2splits[itself]))
            # This basically builds the hyper2hypo for the current concept
            true_object_concepts = [itself]
            for obj in self.objects_:
                ro = [o['related_concept'] for o in self.relations[obj] if o['relation_type'] != 'has_attribute']
                if self.concepts_[itself] in ro:
                    true_object_concepts.append(self.entry2idx_[obj])
            true_object_concepts = sorted(list(valid_object_concepts.intersection(true_object_concepts)))
            train_candidates = random.choices(list(chain.from_iterable(self.object2train_images[c] for c in true_object_concepts)),
                                              k=self.N_SAMPLES)
            relations = [self.relation_entry2idx[r] for r in relation_type]
            for i in range(self.N_SAMPLES):
                metaconcept_text = self.metaconcept_text(related_concepts, relations, itself)
                metaconcept_program =  self.metaconcept_program(related_concepts, relations)
                encoded_metaconcept = self.encode(metaconcept_text,
                                                 metaconcept_program, 'metaconcept')
                exist_statement = self.exist_statement(itself)
                exist_statement_program = self.exist_statement_program(itself)
                encoded_statement = self.encode(exist_statement,
                                                exist_statement_program, 'statement')
                train_image_index = [train_candidates[i]]
                for chosen in [itself, *related_concepts]:
                    if self.concepts_[chosen] not in self.attributes.keys():
                        # Build exist questions for non-abstract concepts
                        # e.g. red, sheltering, red_curve_block
                        exist_question = self.exist_question([chosen])
                        exist_question_program = self.exist_question_program([chosen])
                        # find leaf object concepts that are true for the current concept
                        if chosen in self.objects:
                            true_object_concepts_for_chosen = [chosen]
                        else:
                            true_object_concepts_for_chosen = []
                            for obj in self.objects_:
                                ro = [o['related_concept'] for o in self.relations[obj] if
                                      o['relation_type'] != 'has_attribute']
                                if self.concepts_[chosen] in ro:
                                    true_object_concepts_for_chosen.append(self.entry2idx_[obj])
                        true_object_concepts_for_chosen = sorted(list(valid_object_concepts.intersection(true_object_concepts_for_chosen)))
                        true_val_image_candidates = list(set(list(chain.from_iterable(self.object2train_images[c] for c in true_object_concepts_for_chosen))).difference(set(train_candidates)))
                        false_object_concepts_for_chosen = sorted(list(valid_object_concepts.intersection(set(self.concepts).difference(true_object_concepts_for_chosen))))
                        false_val_image_candidates = list(chain.from_iterable(self.object2train_images[c] for c in false_object_concepts_for_chosen))
                        exist_val_false_images = []

                        if (self.domain == "house_study") and (chosen in self.objects):
                            exist_val_false_images = random.choices(false_val_image_candidates,
                                                                    k=int(num_for_oneshot * 0.6))
                            chosen_rels = self.relations[self.concepts_[chosen]]
                            chosen_color = [r["related_concept"] for r in chosen_rels if r["relation_type"] == "has_color"][0]
                            chosen_affordance = [r["related_concept"] for r in chosen_rels if
                                            r["relation_type"] == "has_affordance"][0]
                            false_objects_with_same_colors = []
                            false_objects_with_same_affordance = []
                            for obj in false_object_concepts_for_chosen:
                                this_rels = self.relations[self.concepts_[obj]]
                                this_color = [r["related_concept"] for r in this_rels if
                                                r["relation_type"] == "has_color"][0]
                                this_affordance = [r["related_concept"] for r in this_rels if
                                                     r["relation_type"] == "has_affordance"][0]
                                if this_color == chosen_color:
                                    false_objects_with_same_colors.append(obj)
                                elif this_affordance == chosen_affordance:
                                    false_objects_with_same_affordance.append(obj)
                            if false_objects_with_same_affordance == []:
                                false_val_image_with_same_affordance = random.choices(false_val_image_candidates,
                                                                        k=int(num_for_oneshot * 0.2))
                            else:

                                false_val_image_candidates_with_same_affordance = list(chain.from_iterable(
                                    self.object2train_images[c] for c in false_objects_with_same_affordance))
                                false_val_image_with_same_affordance = random.choices(
                                    false_val_image_candidates_with_same_affordance,
                                    k=int(num_for_oneshot * 0.2))

                            if false_objects_with_same_colors == []:
                                false_val_image_with_same_color = random.choices(false_val_image_candidates,
                                                                                      k=int(num_for_oneshot * 0.2))
                            else:
                                false_val_image_candidates_with_same_color = list(chain.from_iterable(
                                    self.object2train_images[c] for c in false_objects_with_same_colors))
                                false_val_image_with_same_color = random.choices(false_val_image_candidates_with_same_color,
                                                                        k=int(num_for_oneshot * 0.2))

                            exist_val_false_images = exist_val_false_images + false_val_image_with_same_color + false_val_image_with_same_affordance
                        else:
                            exist_val_false_images = random.choices(false_val_image_candidates,
                                             k=num_for_oneshot)
                        try:
                            exist_val_true_images = random.choices(true_val_image_candidates, k=num_for_oneshot)
                        except:
                            breakpoint()

                        val_images_for_exist_question = exist_val_true_images + exist_val_false_images
                        answers_for_exist_question = [True] * len(exist_val_true_images) + [False] * len(exist_val_false_images)
                        exist_questions = [exist_question] * len(val_images_for_exist_question)
                        exist_question_programs = [exist_question_program] * len(val_images_for_exist_question)

                        val_image_idx = val_images_for_exist_question
                        answers = answers_for_exist_question
                        curr_questions = exist_questions
                        curr_question_programs = exist_question_programs
                    else:
                        # Build is_instance_of questions for attribute concepts
                        true_concepts = [self.entry2idx_[x] for x in self.attributes[self.concepts_[chosen]]]
                        attribute_concepts = [self.entry2idx_[x] for v in self.attributes.values() for x in v]

                        valid_true_concepts = [x for x in true_concepts if self.concept2splits[x] <= self.concept2splits[itself]]
                        false_concepts = set(attribute_concepts).difference(true_concepts)
                        valid_false_concepts = [x for x in false_concepts if self.concept2splits[x] <= self.concept2splits[itself]]
                        selected_true_concepts = random.choices(valid_true_concepts, k = 20)
                        selected_false_concepts = random.choices(valid_false_concepts, k = 20)
                        # for these concepts, questions should be independent of image
                        selected_concepts = selected_true_concepts + selected_false_concepts
                        val_image_idx = [0] * len(selected_concepts)
                        answers = [True] * len(selected_true_concepts) + [False] * len(selected_false_concepts)
                        curr_questions = [self.is_instance_of_question(sc, chosen) for sc in selected_concepts]
                        curr_question_programs = [self.is_instance_of_question_program(sc, chosen) for sc in selected_concepts]
                    for j, (ti, vi, q, qp, a) in enumerate(zip(cycle(train_image_index), val_image_idx, curr_questions, curr_question_programs, answers)):
                        questions.append(
                            {
                                'statement': exist_statement,
                                'statement_program': exist_statement_program,
                                'metaconcept_text': metaconcept_text,
                                'metaconcept_program': metaconcept_program,
                                'question': q, 'question_program': qp, 'answer': a,
                                'concept_index': chosen,
                                'train_image_index': ti, 'image_index': vi,
                                'family': (itself, i, j)
                            }
                        )

        return questions

    # def _build_questions(self):
    #     # This current versions care only about the same branch of the tree
    #     # We may want to include various types of edges in the future
    #     questions = []
    #     # This just avoid using test images
    #     img_candidates = [i for i in range(len(self.image_split_specs)) if self.image_split_specs[i] == 0]
    #     # currently hard-code not to consider inserting new attribute
    #     for itself in tqdm(self.concepts[:-2]):
    #         # we don't want to propogate "has_attribute" relations
    #         # mainly we don't want to adjust the embedding of abstract concepts such as "affordance" and "color" when we are inserting an object
    #         related_concepts = [self.entry2idx_[r['related_concept']] for r in self.relations[self.concepts_[itself]] if r['relation_type'] != 'has_attribute']
    #         relation_type = [r['relation_type'] for r in self.relations[self.concepts_[itself]] if r['relation_type'] != 'has_attribute']
    #         # check the type of current concept by checking whether is_instance_of relation in relation type
    #         itself_type = "object"
    #         for r in self.relations[self.concepts_[itself]]:
    #             if r['relation_type'] == 'is_instance_of':
    #                 itself_type = r['related_concept']
    #                 break
    #         valid_object_concepts = set(nonzero(self.concept2splits[:len(self.objects)] <= self.concept2splits[itself]))
    #         # This basically builds the hyper2hypo for the current concept
    #         true_object_concepts = [itself]
    #         for obj in self.objects_:
    #             ro = [o['related_concept'] for o in self.relations[obj] if o['relation_type'] != 'has_attribute']
    #             if self.concepts_[itself] in ro:
    #                 true_object_concepts.append(self.entry2idx_[obj])
    #         true_object_concepts = sorted(list(valid_object_concepts.intersection(true_object_concepts)))
    #         train_image_candidates = list(set(chain.from_iterable(self.object2train_images[c] for c in true_object_concepts)).intersection(img_candidates))
    #         train_candidates = random.choices(train_image_candidates, k=self.N_SAMPLES)
    #         relations = [self.relation_entry2idx[r] for r in relation_type]
    #         for i in range(self.N_SAMPLES):
    #             metaconcept_text = self.metaconcept_text(related_concepts, relations, itself)
    #             metaconcept_program =  self.metaconcept_program(related_concepts, relations)
    #             encoded_metaconcept = self.encode(metaconcept_text,
    #                                              metaconcept_program, 'metaconcept')
    #             exist_statement = self.exist_statement(itself)
    #             exist_statement_program = self.exist_statement_program(itself)
    #             encoded_statement = self.encode(exist_statement,
    #                                             exist_statement_program, 'statement')
    #             train_image_index = [train_candidates[i]]
    #             for chosen in [itself, *related_concepts]:
    #                 exist_val_false_images = []
    #                 if itself_type == "object":
    #                     true_val_image_candidates = list(set(list(self.object2train_images[itself])).intersection(img_candidates).difference(set(train_candidates)))
    #                     itself_true_images = random.choices(true_val_image_candidates, k=2)
    #                 else:
    #                     itself_true_images = []
    #                 if self.concepts_[chosen] not in self.attributes.keys():
    #                     # Build exist questions for non-abstract concepts
    #                     # e.g. red, sheltering, red_curve_block
    #                     exist_question = self.exist_question([chosen])
    #                     exist_question_program = self.exist_question_program([chosen])
    #                     # find leaf object concepts that are true for the current concept
    #                     if chosen in self.objects:
    #                         true_object_concepts_for_chosen = [chosen]
    #                         object_color = [r["related_concept"] for r in self.relations[self.concepts_[chosen]] if
    #                                         r['relation_type'] == 'has_color'][0]
    #                         object_afford = [r["related_concept"] for r in self.relations[self.concepts_[chosen]] if
    #                                          r['relation_type'] == 'has_affordance'][0]
    #
    #                         negative_concept_candidate_of_same_color = []
    #                         negative_concept_candidate_of_same_affordance = []
    #                         for obj in self.objects_:
    #                             if (obj == self.objects_[chosen]) or (
    #                                     self.concept2splits[self.entry2idx_[obj]] > self.concept2splits[chosen]):
    #                                 continue
    #                             rels = self.relations[obj]
    #                             color = [r['related_concept'] for r in rels if r['relation_type'] == 'has_color'][
    #                                 0]
    #                             affordance = \
    #                             [r['related_concept'] for r in rels if r['relation_type'] == 'has_affordance'][0]
    #                             if color == object_color:
    #                                 negative_concept_candidate_of_same_color.append(self.entry2idx_[obj])
    #                             elif affordance == object_afford:
    #                                 negative_concept_candidate_of_same_affordance.append(self.entry2idx_[obj])
    #                             else:
    #                                 continue
    #                         if (len(negative_concept_candidate_of_same_affordance) != 0) and (len(negative_concept_candidate_of_same_color) != 0):
    #                             false_images_candidates_from_concept_with_same_color = list(set(chain.from_iterable(
    #                                 self.object2train_images[c] for c in negative_concept_candidate_of_same_color)).intersection(
    #                                 img_candidates))
    #                             false_images_candidates_from_concept_with_same_affordance = list(set(chain.from_iterable(
    #                                 self.object2train_images[c] for c in
    #                                 negative_concept_candidate_of_same_affordance)).intersection(
    #                                 img_candidates))
    #                             false_images_from_concept_with_same_color = random.choices(false_images_candidates_from_concept_with_same_color, k=2)
    #                             false_images_from_concept_with_same_affordance = random.choices(
    #                                 false_images_candidates_from_concept_with_same_affordance, k=2)
    #                             exist_val_false_images.extend(false_images_from_concept_with_same_color)
    #                             exist_val_false_images.extend(false_images_from_concept_with_same_affordance)
    #                     else:
    #                         true_object_concepts_for_chosen = []
    #                         for obj in self.objects_:
    #                             ro = [o['related_concept'] for o in self.relations[obj] if
    #                                   o['relation_type'] != 'has_attribute']
    #                             if self.concepts_[chosen] in ro:
    #                                 true_object_concepts_for_chosen.append(self.entry2idx_[obj])
    #
    #
    #                     true_object_concepts_for_chosen = sorted(list(valid_object_concepts.intersection(true_object_concepts_for_chosen)))
    #                     true_val_image_candidates = list(set(list(chain.from_iterable(self.object2train_images[c] for c in true_object_concepts_for_chosen))).intersection(img_candidates).difference(set(train_candidates)))
    #                     false_object_concepts_for_chosen = sorted(list(valid_object_concepts.intersection(
    #                         set(self.concepts).difference(true_object_concepts_for_chosen))))
    #                     false_val_image_candidates = list(set(chain.from_iterable(
    #                         self.object2train_images[c] for c in false_object_concepts_for_chosen)).intersection(
    #                         img_candidates))
    #                     if len(exist_val_false_images) > 0:
    #                         exist_val_false_images.extend(random.choices(false_val_image_candidates, k=6))
    #                     else:
    #                         exist_val_false_images = random.choices(false_val_image_candidates, k=10)
    #                     if len(itself_true_images) > 0:
    #                         exist_val_true_images = itself_true_images
    #                         exist_val_true_images.extend(random.choices(true_val_image_candidates, k=8))
    #                     else:
    #                         exist_val_true_images = random.choices(true_val_image_candidates, k=10)
    #
    #
    #                     val_images_for_exist_question = exist_val_true_images + exist_val_false_images
    #                     answers_for_exist_question = [True] * len(exist_val_true_images) + [False] * len(exist_val_false_images)
    #                     exist_questions = [exist_question] * len(val_images_for_exist_question)
    #                     exist_question_programs = [exist_question_program] * len(val_images_for_exist_question)
    #
    #                     val_image_idx = val_images_for_exist_question
    #                     answers = answers_for_exist_question
    #                     curr_questions = exist_questions
    #                     curr_question_programs = exist_question_programs
    #                 else:
    #                     # Build is_instance_of questions for attribute concepts
    #                     true_concepts = [self.entry2idx_[x] for x in self.attributes[self.concepts_[chosen]]]
    #                     attribute_concepts = [self.entry2idx_[x] for v in self.attributes.values() for x in v]
    #
    #                     valid_true_concepts = [x for x in true_concepts if self.concept2splits[x] <= self.concept2splits[itself]]
    #                     false_concepts = set(attribute_concepts).difference(true_concepts)
    #                     valid_false_concepts = [x for x in false_concepts if self.concept2splits[x] <= self.concept2splits[itself]]
    #                     selected_true_concepts = random.choices(valid_true_concepts, k = 10)
    #                     selected_false_concepts = random.choices(valid_false_concepts, k = 10)
    #                     # for these concepts, questions should be independent of image
    #                     selected_concepts = selected_true_concepts + selected_false_concepts
    #                     val_image_idx = [img_candidates[0]] * len(selected_concepts)
    #                     answers = [True] * len(selected_true_concepts) + [False] * len(selected_false_concepts)
    #                     curr_questions = [self.is_instance_of_question(sc, chosen) for sc in selected_concepts]
    #                     curr_question_programs = [self.is_instance_of_question_program(sc, chosen) for sc in selected_concepts]
    #                 for j, (ti, vi, q, qp, a) in enumerate(zip(cycle(train_image_index), val_image_idx, curr_questions, curr_question_programs, answers)):
    #                     questions.append(
    #                         {
    #                             'statement': exist_statement,
    #                             'statement_program': exist_statement_program,
    #                             'metaconcept_text': metaconcept_text,
    #                             'metaconcept_program': metaconcept_program,
    #                             'question': q, 'question_program': qp, 'answer': a,
    #                             'concept_index': chosen,
    #                             'train_image_index': ti, 'image_index': vi,
    #                             'family': (itself, i, j)
    #                         }
    #                     )
    #
    #     return questions

    def save_inference_handler(self):
        # TODO method bound to change for tabletop domains
        questions = []
        output_dir = f'{self.augmented_root}/{self.name[:-8]}'
        mkdir(output_dir)
        for family, qs in groupby(self.questions, lambda q: (q['family'][:2])):
            qs = sorted(list(qs), key=lambda q: q['family'][2])
            train_texts = [q['statement'] for q in qs[:self.shot_k]]
            train_programs = [self.nscl2program(self.clevr2nscl(q['statement_program'])) for q in qs[:self.shot_k]]
            train_answers = [True] * self.shot_k
            train_image_indices = [q['image_index'] for q in qs[:self.shot_k]]
            metaconcept_text = qs[0]['metaconcept_text']
            train_samples = {'text': train_texts, 'program': train_programs, 'answer': train_answers,
                'image_index': train_image_indices, 'metaconcept_text': metaconcept_text}


            val_text = [q['question'] for q in qs]
            val_program = [self.nscl2program(self.clevr2nscl(q['question_program'])) for q in qs]
            val_answer = [q['answer'] for q in qs]
            val_image_indices = [q['image_index'] for q in qs]
            val_samples = {'text': val_text, 'program': val_program, 'answer': val_answer,
                'image_index': val_image_indices}

            supports = [self.entry2idx_[p['value_inputs'][0]] for p in qs[0]['metaconcept_program']]
            relations = [self.metaconcepts_.index(p['type']) for p in qs[0]['metaconcept_program']]
            text = ' '.join(train_texts + [qs[0]['metaconcept_text']] + val_text)
            text = ' '.join([qs[0]['metaconcept_text']] + val_text)
            concept_index = [q["concept_index"] for q in qs]
            q = {'text': text, 'train_sample': train_samples, 'val_sample': val_samples, 'supports': supports,
                 'relations': relations, 'concept_index': concept_index}
            #q = {'text': text, 'val_sample': val_samples, 'supports': supports,
            #    'relations': relations, 'concept_index': concept_index}
            questions.append(q)
        dest_dataset = self.get_augmented_name(self.__class__.__qualname__).replace('_builder', '')
        #dump(questions, join(output_dir, f"{evaluated['mode']}_{iteration:07d}.json"))
        #dump(questions, join(mkdir(join(self.augmented_root, dest_dataset)), 'questions.json'))
        dump(questions, join(output_dir, f'questions.json'))
    def dump_questions(self):

        self.save_inference_handler()

