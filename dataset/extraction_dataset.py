import torch
import json
import math
from transformers import BertTokenizer, T5Tokenizer
from dataset.utils import sample_with_ratio
from utils import to_cpu_detach, nonzero, load
from torch.utils.data import RandomSampler, SequentialSampler
import random




class ExtractionDataset(torch.utils.data.Dataset):

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
        self.split2spec = {"train": 0, "val": 1}
        self.data = self._load_data()
        self.split = cfg.split
        self.split_specs = self._build_split_specs()
        self.indices_split = self.select_split(self.split_specs)
        self.batch_size = 1




    data_path = "/home/local/ASUAD/weiweigu/Desktop/extraction_dataset.json"


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
        for i in range(10000):
            if i < 7000:
                spec.append(0)
            else:
                spec.append(1)
        return torch.tensor(spec)

    def get_batch_sampler(self, batch_size):
        return None

    @property
    def sampler(self):
        if self.split in ["train", "val"]:
            sampler = RandomSampler(self)
        else:
            sampler = SequentialSampler(self)
        return sampler



    def _load_data(self):
        # load data into tensors
        data = []
        with open(self.data_path, "r") as f:
            data_list = json.load(f)
        for i, d in enumerate(data_list):
            sentence = d["sentence"]
            candidates = d["candidates"]
            label = torch.tensor(d["label"])
            contextualized_sentences = []
            masks = []
            for c in candidates:
                encoded_contextualized_sentence = self.tokenizer.encode_plus(
                    sentence,
                    c,
                    add_special_tokens=True,
                    max_length=512,
                    return_attention_mask=True,
                    pad_to_max_length=True,
                    truncation=True,
                    return_tensors='pt'
                )
                contextualized_sentences.append(encoded_contextualized_sentence['input_ids'])
                masks.append(encoded_contextualized_sentence['attention_mask'])
            contextualized_sentences = torch.stack(contextualized_sentences, dim=1).squeeze(0)
            masks = torch.stack(masks, dim=1).squeeze(0)

            dp = {
                    **d, "contextualized_sentences_tokens": contextualized_sentences,
                    "attention_mask": masks,
                  }
            dp["label"] = label
            data.append(dp)
        return data


    def __getitem__(self, index):
        data_point_index = self.indices_split[index]
        data_point = self.data[data_point_index]
        return data_point
