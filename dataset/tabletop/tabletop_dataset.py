from collections import defaultdict
import copy

import torch
import torchvision.transforms.functional as TF

import random
import numpy as np
import torch

from dataset.dataset import Dataset, BuilderDataset
from dataset.utils import FixedResizeTransform, ProgramVocab
from dataset.utils import sample_with_ratio, WordVocab
from utils import join, load, read_image, file_cached, mask2bbox


class TabletopDataset(Dataset):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        # load concepts and the hierarchy
        self.root = self.root + f"/{cfg.DOMAIN}"
        self.domain = cfg.DOMAIN.lower()
        self._relation_file = f'knowledge/tabletop_{self.domain}_relations.json'
        self.relations = load(self._relation_file)
        self.attributes = {}
        if "house" == self.domain:
            self.attributes = {
                "affordance": ["sheltering", "flooring", "supporting"],
                "color": ["wood", "red", "yellow", "green", "blue", "orange"]
            }
        self.objects_ = self._read_concept_names()
        self.objects = list(range(len(self.objects_)))

        self.concepts_ = self._build_concepts()

        self.concepts = list(range(len(self.concepts_)))
        self.entry2idx_ = {e: i for i, e in enumerate(self.concepts_)}

        self.named_entries_ = self.concepts_
        self.names = [self.named_entries_[_].replace('_', ' ').lower() for _ in self.concepts]
        self.relation_vocab = self._read_relation_names()
        self.relation_entry2idx = {
            x: i for i,x in enumerate(self.relation_vocab)
        }
        #image_information
        self.image_filenames = self._read_image_filenames()
        self.annotation_filenames = self._read_annotation_filenames()
        self.image2object = torch.tensor(self._read_image2object())
        self.object2images, self.object2train_images = self._build_object2images()
        self.image_annotations = self._get_image2annotation()

        # build splits
        self.image_split_specs = torch.tensor(self._build_image_split_specs())
        self.concept_split_specs = torch.tensor(self._build_concept_split_specs())
        if "house" in self.domain:
            self.concept2splits = (self.concept_split_specs - 1).float().div(3).clamp(min=-1.,max=1.).floor()
        else:
            self.concept2splits = (self.concept_split_specs - 1).float().div(4).clamp(min=-1.,max=1.).floor()
        self.has_mask = True

        self.word_vocab = self._build_word_vocab()


    # need image2class && image2difficulty
    # leaf_concepts
    _image2object_file = 'image_object_labels.txt'
    _image_filename_file = 'images.txt'
    _annotation_filename_file = 'annotations.txt'
    _object_file = "objects.txt"
    _relation_name_file = 'relations.txt'

    def _get_image2annotation(self):
        imageidx2annotations = []
        for anno_path in self.annotation_filenames:
            annotation = {
                "objects" : []
            }
            curr_anno = load(f"{self.root}/annotations/{anno_path}")
            for obj in curr_anno["shapes"]:
                object = {"label": obj["label"]}
                points = obj["points"]
                x_s = [x[0] for x in points]
                y_s = [x[1] for x in points]
                bounding_box = [[min(x_s), max(x_s)], [min(y_s), max(y_s)]]
                object["bounding_box"] = bounding_box
                annotation['objects'].append(object)
            imageidx2annotations.append(annotation)
        return imageidx2annotations




    def _read_names(self, path):
        with open(join(self.root, path), 'r') as f:
            read_lines = f.readlines()
        return [line.rstrip('\n') for line in read_lines]

    @file_cached(_object_file)
    def _read_concept_names(self):
        return self._read_names(self._object_file)

    @file_cached(_relation_name_file)
    def _read_relation_names(self):
        return self._read_names(self._relation_name_file)


    @file_cached('concepts')
    def _build_concepts(self):
        concepts = list(self.relations.keys())
        return concepts

    @file_cached(_image2object_file)
    def _read_image2object(self):
        return [self.entry2idx_[x.split("/")[0]] for x in self.image_filenames]

    def _build_object2images(self):
        object2images = defaultdict(list)
        for i, c in enumerate(self.image2object):
            object2images[c.item()].append(i)
        # train image shall be images that contain only the object itself
        # right now assume that we only use that type of images
        object2train_images = copy.deepcopy(object2images)
        return object2images, object2train_images

    @file_cached(_annotation_filename_file)
    def _read_annotation_filenames(self):
        with open(join(self.root, self._annotation_filename_file), 'r') as f:
            read_lines = f.readlines()
        return [line.split(' ')[-1].rstrip('\n') for line in read_lines]

    @file_cached(_image_filename_file)
    def _read_image_filenames(self):
        with open(join(self.root, self._image_filename_file), 'r') as f:
            read_lines = f.readlines()
        return [line.split(' ')[-1].rstrip('\n') for line in read_lines]



    @file_cached("image_split")
    def _build_image_split_specs(self):
        split_specs = []
        for c, image_indices in self.object2images.items():
            split_specs.append(sample_with_ratio(len(image_indices), self.split_ratio, 10))
        return torch.cat(split_specs).tolist()


    @file_cached("concept_split")
    def _build_concept_split_specs(self):
        random.seed(self.split_seed)
        torch.manual_seed(self.split_seed)
        np.random.seed(self.split_seed)
        manual_seed = []
        if self.domain == "house":
            manual_seed = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14,100, 101, 102, 103, 104, 105, 999]
        elif "house" in self.domain:
            manual_seed = [0, 1, 2]
        manual_splits = {
            "house_study": {
                0: {
                    "pretrain": [0, 5, 8, 9, 11, 14],
                    "val": [3, 7, 10, 15]
                },
                1: {
                    "pretrain": [],
                    "val": [3, 7, 10, 15]
                },
                2: {
                    "pretrain": [],
                    "val": [3, 7, 10, 15]
                }
            },
            "house": {
                0: {
                    "pretrain": [0, 7, 15, 16, 24, 28],
                    "val": [5, 9, 12, 17, 23, 27]
                },
                1: {
                    "pretrain": [1, 6, 12, 19, 23, 30],
                    "val": [2, 10, 11, 16, 24, 28]
                },
                2: {
                    "pretrain": [4, 8, 11, 16, 22, 30],
                    "val": [3, 9, 12, 17, 24, 29]
                },
                3: {
                    "pretrain": [2, 8, 15, 19, 21, 26],
                    "val": [1, 6, 13, 20, 22, 30]
                },
                4: {
                    "pretrain": [0, 9, 13, 20, 21, 27],
                    "val": [2, 8, 14, 16, 25, 26]
                },
                10: {
                    "pretrain": [],
                    "val": [5, 9, 12, 17, 23, 27]
                },
                11: {
                    "pretrain": [],
                    "val": [2, 10, 11, 16, 24, 28]
                },
                12: {
                    "pretrain": [],
                    "val": [3, 9, 12, 17, 24, 29]
                },
                13: {
                    "pretrain": [],
                    "val": [1, 6, 13, 20, 22, 30]
                },
                14: {
                    "pretrain": [],
                    "val": [2, 8, 14, 16, 25, 26]
                },
                100: {
                    # "pretrain": [0, 7, 15, 16, 24, 28],
                    # "val": [5, 9, 12, 17, 23, 27]
                    "pretrain": [0, 4, 6, 7, 12, 15, 16, 19, 23, 25, 28, 30],
                    "val": [5, 9, 13, 17, 22]
                },
                101: {
                    "pretrain": [1, 3, 6, 9, 12, 13, 17, 19, 23, 26, 27, 30],
                    "val": [2, 10, 11, 16, 24]
                },
                102: {
                    "pretrain": [1, 4, 6, 10, 11, 14, 18, 19, 21, 25, 27, 29],
                    "val": [3, 9, 12, 16, 24, 30]
                },
                103: {
                    "pretrain": [0, 2, 7, 8, 11, 15, 17, 19, 21, 24, 26, 28],
                    "val": [1, 6, 13, 20, 22, 30]
                },
                104: {
                    "pretrain": [0, 1, 7, 9, 11, 13, 17, 20, 21, 24, 27, 30],
                    "val": [2, 8, 14, 16, 25, 26]
                },
                105: {
                    "pretrain": [0, 7, 15, 16, 24, 28],
                    "val": [5, 9, 12, 17, 23, 27]

                },
                999: {
                    #"pretrain": [0, 7, 15, 16, 24, 28],
                    #"val": [5, 9, 12, 17, 23, 27]
                    "pretrain": [0, 4, 6, 7, 12, 15, 16, 19, 23, 25, 28, 30],
                    "val": [5, 9, 13, 17, 22]
                }
            },
            "park": {

            }
        }
        if self.split_seed in manual_seed:
            splits = manual_splits[self.domain][self.split_seed]
            pretrain_leaf_concepts = set(splits["pretrain"])
            val_leaf_concepts = set(splits["val"])
            train_leaf_concepts = set(self.objects).difference(pretrain_leaf_concepts).difference(val_leaf_concepts)
        else:
            #temp_split_range = list(range(len(self.objects_)))
            temp_split_range = list(range(5))
            temp_split_range.remove(self.split_seed % 10)
            random.shuffle(temp_split_range)
            if self.split_seed in list(range(5)):
                split_remainder = [temp_split_range[0:2], temp_split_range[2:], [self.split_seed % 10]]
            elif self.split_seed in list(range(10, 15)):
                split_remainder = [[temp_split_range[0]], temp_split_range[1:], [self.split_seed % 10]]
            else:
                split_remainder = [[], temp_split_range, [self.split_seed % 10]]
            #split_remainder = [temp_split_range[:2], temp_split_range[2:], [self.split_seed]]

            pretrain_remainder, train_remainder, val_remainder = split_remainder
            #pretrain_leaf_concepts = set(c for i,c in enumerate(temp_split_range) if i % 5 in pretrain_remainder)
            #train_leaf_concepts = set(c for i,c in enumerate(temp_split_range) if i % 5 in train_remainder)
            #val_leaf_concepts = set(c for i,c in enumerate(temp_split_range) if i % 5 in val_remainder)
            pretrain_leaf_concepts = set(c for c in self.concepts if c % 5 in pretrain_remainder)
            train_leaf_concepts = set(c for c in self.concepts if c % 5 in train_remainder)
            val_leaf_concepts = set(c for c in self.concepts if c % 5 in val_remainder)

        pretrain_concepts, train_concepts, val_concepts = set(pretrain_leaf_concepts), set(train_leaf_concepts), set(val_leaf_concepts)

        for pretrain_leaf_concept in pretrain_leaf_concepts:
            c = self.concepts_[pretrain_leaf_concept]
            relations = self.relations[c]
            related_concepts = [self.entry2idx_[r['related_concept']] for r in relations]
            pretrain_concepts.update(related_concepts)
        for train_leaf_concept in train_leaf_concepts:
            c = self.concepts_[train_leaf_concept]
            relations = self.relations[c]
            related_concepts = [self.entry2idx_[r['related_concept']] for r in relations]
            train_concepts.update(related_concepts)
        train_concepts.difference_update(pretrain_concepts)
        for val_leaf_concept in val_leaf_concepts:
            c = self.concepts_[val_leaf_concept]
            relations = self.relations[c]
            related_concepts = [self.entry2idx_[r['related_concept']] for r in relations]
            val_concepts.update(related_concepts)
        val_concepts.difference_update(pretrain_concepts)
        val_concepts.difference_update(train_concepts)

        split_spec = [-100] * len(self.concepts)
        if "house" in self.domain:
            for c in pretrain_concepts:
                split_spec[c] = -4
            for c in train_concepts:
                split_spec[c] = 0
            for c in val_concepts:
                split_spec[c] = 4
            for c, r in self.relations.items():
                r = [x['relation_type'] for x in r]
                c = self.entry2idx_[c]
                if ('has_color' in r) or ('has_affordance' in r):
                    split_spec[c] += 2
                else:
                    split_spec[c] += 1
        else:
            for c in pretrain_concepts:
                split_spec[c] = -4
            for c in train_concepts:
                split_spec[c] = 1
            for c in val_concepts:
                split_spec[c] = 6
            for c, r in self.relations.items():
                r = [x['relation_type'] for x in r]
                c = self.entry2idx_[c]
                if 'great_grandparent' in r:
                    split_spec[c] += 3
                elif 'grandparent' in r:
                    split_spec[c] += 2
                elif 'parent' in r:
                    split_spec[c] += 1
        return split_spec

    @property
    def transform_fn(self):
        return FixedResizeTransform

    def get_stacked_scenes(self, image_index):
        assert not torch.is_tensor(image_index)
        img = self.get_image(image_index)
        if not self.has_mask:
            return {"image": self.transform(img)}
        else:
            mask = self.get_mask(image_index)
            mask, img = self.transform(mask, img)
            return {"image": img, "mask": mask}

    def get_mask(self, image_index):
        # now that assuming we have only one object in the image
        annotation = self.image_annotations[image_index]
        obj = annotation['objects'][0]
        bounding_box = obj['bounding_box']
        image = self.get_image(image_index)
        mask = torch.zeros(size=image.shape[1:])
        for h in range(mask.shape[0]):
            for w in range(mask.shape[1]):
                if (w >= bounding_box[0][0]) and (w <= bounding_box[0][1]) and (h >= bounding_box[1][0]) and (h <= bounding_box[1][1]):
                    mask[h, w] = 1
        mask = (mask > 0).unsqueeze(0)
        return mask

    # Computed
    def get_image(self, image_index):
        return TF.to_tensor(read_image(join(self.root, "images", self.image_filenames[image_index])))

    def get_annotation_by_image_idx(self, image_index):
        annotation = load(join(self.root, self.annotation_filenames[image_index]))
        return annotation

    def unique_question(self, candidates):
        if len(candidates) > 1:
            names = [self.names[c] for c in candidates]
            name = ", ".join(names)
            return f"Which is the {name} object?"
        else:
            candidates = candidates[0]
            name = self.concepts_[candidates]
            if name in self.objects_:
                return f"Which is the {self.names[candidates]}?"
            else:
                return f"Which is the {self.names[candidates]} object?"

    def exist_question(self, candidates):
        if len(candidates) > 1:
            names = [self.concepts_[c] for c in candidates]
            name = ", ".join(names)
            return f"Is there a {name} object?"
        else:
            candidates = candidates[0]
            name = self.concepts_[candidates]
            if name in self.objects_:
                return f"Is there a {self.names[candidates]}?"
            else:
                return f"Is there a {self.names[candidates]} object?"

    def exist_statement(self, candidate):
        name = self.concepts_[candidate]
        if name in self.objects_:
            return f"There is a {self.names[candidate]}."
        else:
            return f"There is a {self.names[candidate]} object."

    def metaconcept_text(self, supports, relations, concept_index):
        if len(supports) == 0:
            return ""
        if "house" in self.domain:
            # 0: index for has_affordance.
            # 1: index for has_color.
            # 2: index for is_instance_of.
            # 0 and 1 are only used for leaf(object) concepts
            # 2 is used for color concepts (e.g. red) and affordance concept(strictly associated with shapes, e.g. sheltering(curved_blocks))
            has_affordances = list(self.names[s] for e, s in zip(relations, supports) if e == 0)
            has_color = list(self.names[s] for e, s in zip(relations, supports) if e == 1)
            #
            is_type = list(self.names[s] for e, s in zip(relations, supports) if e == 2)
            assert ((len(has_affordances) == 0) or (len(is_type) == 0)), "Can't have a abstract concept without an affordance!"
            if len(is_type) != 0:
                assert len(is_type) == 1, "A concept can't have more than one type!"
                return f"{self.names[concept_index]} is a {is_type[0]}."
            else:
                sent = ""
                if len(has_color) != 0:
                    sent += f"{self.names[concept_index]} is of color {has_color[0]}. "
                names = ", ".join(has_affordances)
                sent += f"{self.names[concept_index]} has affordances of {names}."
                return sent
        else:
            text = []
            meta_relations = ["parent", "grandparent", "great grandparent"]
            for r, s in zip(relations, supports):
                sentence = f"{self.names[s]} is the {meta_relations[r]} of {self.names[concept_index]}."
                text.append(sentence)
            return " ".join(text)


    def is_instance_of_question(self, concept_1, concept_2):
        return f"Is {self.names[concept_1]} an instance of {self.names[concept_2]}?"

    #@file_cached("word_tokens")
    def _build_word_tokens(self):
        vocabulary = WordVocab()
        vocabulary.update(self.names)
        vocabulary.update(["of"])
        vocabulary.update([self.exist_statement(0), self.exist_question([0]), self.unique_question([0]), self.metaconcept_text([0], [2], 0),self.metaconcept_text([0, 0], [0, 1], 0), self.is_instance_of_question(0, 0)])

        vocabulary.update(['yes', 'no', 'object'])
        return sorted(list(vocabulary.words))


class TabletopBuilderDataset(BuilderDataset, TabletopDataset):
    num_inputs = {'scene': 0, 'filter': 1, 'exist': 1, 'unique': 1, 'is_instance_of': 0}
    num_value_inputs = {'scene': 0, 'filter': 1, 'exist': 0, 'unique': 0, 'is_instance_of': 2}
    assert num_inputs.keys() == num_value_inputs.keys()

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.metaconcepts_ = self.relation_vocab
        self.program_vocab = self._build_program_vocab()


    @file_cached('program_tokens')
    def _build_program_tokens(self):
        vocabulary = ProgramVocab()
        vocabulary.update(self.named_entries_)
        vocabulary.update(self.num_inputs.keys())
        vocabulary.update(self.metaconcepts_)
        return sorted(list(vocabulary.words))

    def exist_question_program(self, candidates):
        prog = [{'type': 'scene', 'inputs': [], 'value_inputs': []}]
        while len(candidates) != 0:
            candidate = candidates[0]
            index = len(prog) - 1
            prog.append({'type': 'filter', 'inputs': [index], 'value_inputs': [self.named_entries_[candidate]]})
            candidates = candidates[1:]
        index = len(prog) - 1
        prog.append({'type': 'exist', 'inputs': [index], 'value_inputs': []})
        return prog

    def is_instance_of_question_program(self, concept_1, concept_2):
        prog = [{'type': 'is_instance_of', 'inputs': [], 'value_inputs': [self.named_entries_[concept_1], self.named_entries_[concept_2]]}]
        return prog