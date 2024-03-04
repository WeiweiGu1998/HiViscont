import os
import random
import copy
from itertools import chain, cycle, groupby, cycle
import math
import torch
from tqdm import tqdm

from dataset.cub.cub_dataset import CubDataset, CubBuilderDataset
from dataset.meta_dataset import MetaDataset, MetaBuilderDataset
from dataset.utils import sample_with_ratio
from utils import file_cached, join, nonzero, mkdir, dump,  to_serializable


class CubFewshotDataset(CubDataset, MetaDataset):

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
            dump(to_serializable(metrics), join(output_dir, f"{filename_prefix}_{iteration:07d}.json"))


class CubDetachedDataset(CubFewshotDataset):
    pass


class CubZeroshotDataset(CubFewshotDataset):
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


class CubFewshotBuilderDataset(CubBuilderDataset, MetaBuilderDataset):
    N_SAMPLES = 5

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self._build_mac()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    @file_cached('questions')
    def _build_questions(self):
        questions = []
        for itself in tqdm(self.concepts[:-1]):
            hyper = min(self.hypo2hyper[itself])
            samekinds = [s for s in self.itself2samekinds[itself] if
                self.concept_split_specs[s] < max(self.concept_split_specs[itself], 1)]
            valid_classes = set(nonzero(self.concept2splits[:len(self.classes)] == self.concept2splits[itself]))
            true_classes = list(sorted(valid_classes.intersection([itself, *self.hyper2hypo[itself]])))
            true_candidates = list(chain.from_iterable(self.class2images[c] for c in true_classes))
            false_classes = list(sorted(valid_classes.difference([itself, *self.hyper2hypo[itself]])))
            false_candidates = list(chain.from_iterable(self.class2images[c] for c in false_classes))
            for i in range(self.N_SAMPLES):
                this_samekinds = self.dropout(samekinds, self.concept_split_specs[itself] <= 0)
                supports = [hyper, *this_samekinds]
                relations = [0] + [2] * len(this_samekinds)
                encoded_metaconcept = self.encode(self.metaconcept_text(supports, relations, itself),
                    self.metaconcept_program(supports, relations), 'metaconcept')
                encoded_statement = self.encode(self.exist_statement(itself),
                    self.exist_statement_program(itself), 'statement')
                encoded_question = self.encode(self.exist_question(itself), self.exist_question_program(itself),
                    'question')
                true_image_index = random.choices(true_candidates, k=self.query_k // 2 + self.shot_k)
                train_image_index, true_image_index = true_image_index[:self.shot_k], true_image_index[
                self.shot_k:]
                false_image_index = random.choices(false_candidates, k=self.query_k // 2)
                val_image_index = true_image_index + false_image_index
                answers = [True] * len(true_image_index) + [False] * len(false_image_index)
                for j, (ti, vi, answer) in enumerate(zip(cycle(train_image_index), val_image_index, answers)):
                    questions.append(
                        {**encoded_statement, **encoded_metaconcept, **encoded_question, 'answer': answer,
                            'concept_index': itself, 'train_image_index': ti, 'image_index': vi,
                            'family': (itself, i, j)})
        return questions


    @file_cached('mac')
    def _build_mac(self):
        super()._build_mac()

    def mac_split(self, concept_index):
        if self.concept_split_specs[concept_index] <= 0:
            return 'train'
        elif self.concept_split_specs[concept_index] == 1:
            return 'val'
        elif self.concept_split_specs[concept_index] == 6:
            return 'test'
        else:
            return None

class CubFewshotSpecificTestBuilderDataset(CubBuilderDataset, MetaBuilderDataset):
    N_SAMPLES = 5

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()
        self._build_mac()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    @file_cached('questions')
    def _build_questions(self):
        questions = []
        target_concepts = [152, 314, 352, 359]
        # We care only about for the concepts Aves
        hyper = 365

        for itself in tqdm(self.concepts[:-1]):
            hyper = min(self.hypo2hyper[itself])
            samekinds = [s for s in self.itself2samekinds[itself] if
                self.concept_split_specs[s] < max(self.concept_split_specs[itself], 1)]
            valid_classes = set(nonzero(self.concept2splits[:len(self.classes)] == self.concept2splits[itself]))
            true_classes = list(sorted(valid_classes.intersection([itself, *self.hyper2hypo[itself]])))
            true_candidates = list(chain.from_iterable(self.class2images[c] for c in true_classes))
            false_classes = list(sorted(valid_classes.difference([itself, *self.hyper2hypo[itself]])))
            false_candidates = list(chain.from_iterable(self.class2images[c] for c in false_classes))
            for i in range(self.N_SAMPLES):
                this_samekinds = self.dropout(samekinds, self.concept_split_specs[itself] <= 0)
                supports = [hyper, *this_samekinds]
                relations = [0] + [2] * len(this_samekinds)
                encoded_metaconcept = self.encode(self.metaconcept_text(supports, relations, itself),
                    self.metaconcept_program(supports, relations), 'metaconcept')
                encoded_statement = self.encode(self.exist_statement(itself),
                    self.exist_statement_program(itself), 'statement')
                encoded_question = self.encode(self.exist_question(itself), self.exist_question_program(itself),
                    'question')
                true_image_index = random.choices(true_candidates, k=self.query_k // 2 + self.shot_k)
                train_image_index, true_image_index = true_image_index[:self.shot_k], true_image_index[
                self.shot_k:]
                false_image_index = random.choices(false_candidates, k=self.query_k // 2)
                val_image_index = true_image_index + false_image_index
                answers = [True] * len(true_image_index) + [False] * len(false_image_index)
                for j, (ti, vi, answer) in enumerate(zip(cycle(train_image_index), val_image_index, answers)):
                    questions.append(
                        {**encoded_statement, **encoded_metaconcept, **encoded_question, 'answer': answer,
                            'concept_index': itself, 'train_image_index': ti, 'image_index': vi,
                            'family': (itself, i, j)})
        return questions

class CubFewshotHierarchyBuilderDataset(CubBuilderDataset, MetaBuilderDataset):
    N_SAMPLES = 5

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.questions = self._build_questions()

        self.split_specs = sample_with_ratio(len(self.questions), [.2, .1, .7], self.split_seed)
        self.indices_split = self.select_split(self.split_specs)

    def metaconcept_text(self, supports, relations, concept_index):
        text = []
        meta_relations = ["parent", "grandparent", "great grandparent", "great great grandparent"]
        for r, s in zip(relations, supports):
            sentence = f"{self.names[s]} is the {meta_relations[r]} of {self.names[concept_index]}."
            text.append(sentence)
        return " ".join(text)

    def _build_questions(self):
        # This current versions care only about the same branch of the tree
        # We may want to include various types of edges in the future
        questions = []
        # This just avoid using test images
        img_candidates = [i for i in range(len(self.image_split_specs)) if self.image_split_specs[i] == 0]
        used_concepts = self.concepts[:-1]
        if self.split_seed in list(range(100, 105)):
            used_concepts = self.concepts
        for itself in tqdm(used_concepts):
        # for itself in tqdm(self.concepts):
        # deal with one-shotting the root concept for the ablation study later
        # this need to be 
            hypers = self.hypo2hyper[itself]
            supports = copy.deepcopy(hypers)
            #hypers.remove(365)
            supports.sort()
            # compute new embedding for ancestors based on distance
            # use different sets of weights
            relations = list(range(len(supports)))
            #samekinds = [s for s in self.itself2samekinds[itself] if
            #    self.concept_split_specs[s] < max(self.concept_split_specs[itself], 1)]
            valid_classes = set(nonzero(self.concept2splits[:len(self.classes)] <= self.concept2splits[itself]))
            true_classes = list(sorted(valid_classes.intersection([itself, *self.hyper2hypo[itself]])))

            train_image_candidates = list(set(chain.from_iterable(self.class2images[c] for c in true_classes)).intersection(img_candidates))
            try:
                train_candidates = random.choices(train_image_candidates, k=self.N_SAMPLES)
            except:
                breakpoint()
            #false_classes = list(sorted(valid_classes.difference([itself, *self.hyper2hypo[itself]])))
            #false_candidates = list(chain.from_iterable(self.class2images[c] for c in false_classes))
            for i in range(self.N_SAMPLES):
                # each concept has 5 samples
                encoded_metaconcept = self.encode(self.metaconcept_text(supports, relations, itself),
                    self.metaconcept_program(supports, relations), 'metaconcept')
                encoded_statement = self.encode(self.exist_statement(itself),
                    self.exist_statement_program(itself), 'statement')
                train_image_index = [train_candidates[i]]
                # build questions
                for chosen in [itself, *hypers]:
                    # for now, create same number of instances for each concept
                    encoded_question = self.encode(self.exist_question(chosen), self.exist_question_program(chosen),
                        'question')
                    true_classes = list(sorted(valid_classes.intersection([chosen, *self.hyper2hypo[chosen]])))
                    true_candidates = list(set(list(chain.from_iterable(self.class2images[c] for c in true_classes))).intersection(img_candidates).difference(set(train_candidates)))
                    true_image_index = random.choices(true_candidates, k=10)
                    false_classes = list(sorted(valid_classes.difference([chosen, *self.hyper2hypo[chosen]])))
                    if len(false_classes) != 0:
                        # the root node of bird has no negative sample
                        false_candidates = list(set(chain.from_iterable(self.class2images[c] for c in false_classes)).intersection(img_candidates))
                        #true_image_index = random.choices(true_candidates, k=self.query_k // (2 * len([itself, *hypers])))
                        #false_image_index = random.choices(false_candidates, k=self.query_k // (2 * len([itself, *hypers])))
                        false_image_index = random.choices(false_candidates, k=10)
                    else:
                        false_image_index = []
                    val_image_index = true_image_index + false_image_index
                    answers = [True] * len(true_image_index) + [False] * len(false_image_index)
                    for j, (ti, vi, answer) in enumerate(zip(cycle(train_image_index), val_image_index, answers)):
                        questions.append(
                            {**encoded_statement, **encoded_metaconcept, **encoded_question, 'answer': answer,
                                'concept_index': chosen, 'train_image_index': ti, 'image_index': vi,
                                'family': (itself, i, j)})
        return questions

    def save_inference_handler(self, evaluated):
        questions = []
        for family, qs in groupby(evaluated['questions'], lambda q: (q['family'][:2])):
            qs = sorted(list(qs), key=lambda q: q['family'][2])
            train_texts = [q['statement'] for q in qs[:self.shot_k]]
            train_programs = [q['statement_program'] for q in qs[:self.shot_k]]
            train_answers = [True] * self.shot_k
            train_image_indices = [q['image_index'] for q in qs[:self.shot_k]]
            metaconcept_text = qs[0]['metaconcept']
            train_samples = {'text': train_texts, 'program': train_programs, 'answer': train_answers,
                'image_index': train_image_indices, 'metaconcept_text': metaconcept_text}

            val_text = [q['question'] for q in qs]
            val_program = [q['question_program'] for q in qs]
            val_answer = [q['answer'] for q in qs]
            val_image_indices = [q['image_index'] for q in qs]
            val_samples = {'text': val_text, 'program': val_program, 'answer': val_answer,
                'image_index': val_image_indices}

            supports = [self.entry2idx_[p['value_inputs'][0]] for p in qs[0]['metaconcept_program']]
            relations = [self.metaconcepts_.index(p['type']) for p in qs[0]['metaconcept_program']]

            text = ' '.join(train_texts + [qs[0]['metaconcept']] + val_text)
            text = ' '.join([qs[0]['metaconcept']] + val_text)
            concept_index = [q["concept_index"] for q in qs]
            q = {'text': text, 'train_sample': train_samples, 'val_sample': val_samples, 'supports': supports,
                 'relations': relations, 'concept_index': concept_index}
            #q = {'text': text, 'val_sample': val_samples, 'supports': supports,
            #    'relations': relations, 'concept_index': concept_index}
            questions.append(q)
        dest_dataset = self.get_augmented_name(self.__class__.__qualname__).replace('_builder', '')
        #dump(questions, join(output_dir, f"{evaluated['mode']}_{iteration:07d}.json"))
        dump(questions, join(mkdir(join(self.augmented_root, dest_dataset)), 'questions.json'))
        #dest_folder = f"{self.augmented_root}/{dest_dataset}"
        #breakpoint()
        #dump(questions, join(dest_folder, 'cub_fewshot_questions.json'))
    def dump_questions(self):
        outputs = {
            "statement_predicted": [torch.tensor(q['statement_target']) for q in self.questions],
            "metaconcept_predicted": [torch.tensor(q['metaconcept_target']) for q in self.questions],
            "question_predicted": [torch.tensor(q['question_target']) for q in self.questions]
        }
        evaluated = self.batch_inference_handler(self.questions, outputs)
        self.save_inference_handler(evaluated)
