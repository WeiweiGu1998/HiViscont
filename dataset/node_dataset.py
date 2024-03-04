import torch
import json
import math
from transformers import BertTokenizer, T5Tokenizer
from dataset.utils import sample_with_ratio
from utils import to_cpu_detach, nonzero, load
from torch.utils.data import RandomSampler, SequentialSampler
import random


class NodeBatchSampler:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.question_indices = list(range(len(dataset.indices_split)))
        random.shuffle(self.question_indices)

    def __iter__(self):
        for i in range(0, len(self.question_indices), self.batch_size):
            yield self.question_indices[i:i+self.batch_size]

    def __len__(self):
        return math.ceil(len(self.question_indices) / self.batch_size)


class NodeClassifierDataset(torch.utils.data.Dataset):

    split2spec = {"train": 0, "val": 1}
    def __init__(self, cfg):
        if "t5" in cfg.model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(
                cfg.model_name
            )
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                cfg.model_name
            )
        # self.concept_vocab = self._build_concepts()
        # self.special_concept_tokens = list([f"<{x}>" for x in self.concept_vocab.keys()])
        # self.tokenizer.add_special_tokens({
        #     "additional_special_tokens": self.special_concept_tokens
        # })
        self.split_seed = 103
        self.data = self._load_data()
        self.split = cfg.split
        concept_path = f"/home/local/ASUAD/weiweigu/data/test_dataset/.augmented/House/{self.split_seed}/tabletop/concepts.json"
        self.concepts_ = load(concept_path)
        self.entry2idx = {x:i for i,x in enumerate(self.concepts_)}
        concept_split_path = f"/home/local/ASUAD/weiweigu/data/test_dataset/.augmented/House/{self.split_seed}/tabletop/concept_split.json"
        self.concept_split_specs = torch.tensor(load(concept_split_path))
        self.concept2split = (self.concept_split_specs - 1).float().div(3).clamp(min=0.,max=1.).floor()
        self.split_specs = self._build_split_specs()
        self.indices_split = self.select_split(self.split_specs)




    #data_path = ""
    data_path = "/home/local/ASUAD/weiweigu/Desktop/small_dataset.json"
    #concept_path = "knowledge/tabletop_hierarchy.json"


    def select_split(self, split_spec):
        if self.split == "all":
            return list(range(len(split_spec)))
        else:
            return nonzero(split_spec == self.split2spec[self.split])

    def log_info(self):
        pass
    def __len__(self):
        return len(self.indices_split)


    def _build_split_specs(self):
        spec = []
        for d in self.data:
            concept_idx = self.entry2idx[d["query_concept"]]
            spec.append(self.concept2split[concept_idx])
        return torch.tensor(spec)

    def get_batch_sampler(self, batch_size):
        #return NodeBatchSampler(self, batch_size)
        return None

    @property
    def sampler(self):
        if self.split in ["train", "val"]:
            sampler = RandomSampler(self)
        else:
            sampler = SequentialSampler(self)
        return sampler


    # def replace_concept_with_special_concept_token(self, s):
    #     concepts = list(self.concept_vocab.keys())
    #     for i, c in enumerate(concepts):
    #         temp = c.replace("_", " ")
    #         if temp in s:
    #             s=s.replace(temp, self.special_concept_tokens[i])
    #     return s

    def _load_data(self):
        # load data into tensors
        data = []
        with open(self.data_path, "r") as f:
            data_list = json.load(f)
        for i, d in enumerate(data_list):
            # These two lines replaces concept in the sentence with special tokens
            # node_context = self.replace_concept_with_special_concept_token(d["node_context"])
            # instruction = self.replace_concept_with_special_concept_token(d["instruction"])
            node_context = d["node_context"]
            query = d["query"]

            node_concept_name = d["node_concept"]
            query_concept_name = d["query_concept"]
            label = torch.tensor(d["label"])

            encoded_contextualized_query = self.tokenizer.encode_plus(
                node_context,
                query,
                add_special_tokens=True,
                max_length=512,
                return_attention_mask=True,
                pad_to_max_length=True,
                truncation=True,
                return_tensors='pt'
            )

            dp = {
                    **d, "contextualized_query_tokens": encoded_contextualized_query['input_ids'],
                    "contextualized_query_attention_mask": encoded_contextualized_query['attention_mask'],
                  }
            dp["label"] = label
            data.append(dp)
        return data

    # def find_index_for_special_token_from_seq(self, seq, special_token):
    #     # Shape: (1, 1, seq_len)
    #     token_list = []
    #     if special_token == "<empty>":
    #         special_token = '[SEP]'
    #     seq = seq.squeeze().tolist()
    #     for i, t in enumerate(seq):
    #         if self.tokenizer.convert_ids_to_tokens(t) == special_token:
    #             return i
    #         token_list.append(self.tokenizer.convert_ids_to_tokens(t))
    #     breakpoint()
    #     raise AssertionError


    # def find_concept_from_sentence(self, sentence, original_concept_name=None):
    #     concepts = list(self.concept_vocab.keys())
    #     candidate_concepts = []
    #     start_idxs = []
    #     for i, c in enumerate(concepts):
    #         if c == original_concept_name:
    #             continue
    #         special_token_name = self.special_concept_tokens[i]
    #         if special_token_name in sentence:
    #             return c
    #     return "empty"

    def __getitem__(self, index):
        data_point_index = self.indices_split[index]
        data_point = self.data[data_point_index]
        return data_point