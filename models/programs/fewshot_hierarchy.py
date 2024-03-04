from collections import defaultdict

import torch

from .fewshot import Fewshot
from .symbolic import SymbolicProgram
from utils import copy_dict

class FewshotHierarchy(SymbolicProgram):

    def __init__(self, *arg):
        # This class does not perform any actual computation!!!
        # gt_embeddings: support concepts
        # relations: relations of the support concepts with the target concept
        # concept_index: target concept index
        super().__init__(*arg)
        self.support_concepts, self.relations, self.concept_index = arg
        self.training_features = None
        self.support_embeddings = None
        self.concept_embedding = None
        # These variables are used to query the correct embedding in evaluation
        self.validation_concepts = None

    def to_evaluate(self, box_registry, **kwargs):
        # This function converts the current program to executable
        k = {}
        # keep this part as the same
        for group in ["train_sample", "val_sample"]:
            k[group] = [q.evaluate(box_registry, features=f, relations=r) for q, f, r in
                zip(kwargs[group].pop("program"), kwargs[group].pop("features"),
                    kwargs[group].pop("relations"))]
        # kinda pseudo in this part
        # get the embeddings from the support concept indices
        self.kwargs = copy_dict(kwargs)
        for group in ["train_sample", "val_sample"]:
            self.kwargs[group]["program"] = k[group]
        if len(self.support_concepts) > 0:
            # Shape: (num_supports)
            self.support_concepts = torch.tensor(self.support_concepts).to(box_registry.device)
            # Shape: (num_supports, 2 * box_dimension)
            self.support_embeddings = box_registry[self.support_concepts]
            # Shape: (num_supports)
            self.relations = torch.tensor(self.relations).to(box_registry.device)
        else:
            self.support_concepts = None
            self.support_embeddings = None
            self.relations = None
        # Shape: (2 * box_dimension)
        self.concept_embedding = box_registry[torch.tensor(self.concept_index).to(box_registry.device)]
        self.validation_concepts = kwargs["concept_index"]





    @property
    def hypernym_embeddings(self):
        return self.gt_embeddings[list(i for i, r in
        enumerate(self.relations if not torch.is_tensor(self.relations) else self.relations.tolist()) if
        r == 0)]

    @property
    def samekind_embeddings(self):
        return self.gt_embeddings[list(i for i, r in
        enumerate(self.relations if not torch.is_tensor(self.relations) else self.relations.tolist()) if
        r == 2)]

    @property
    def is_fewshot(self):
        return not any(i == -1 for i in self.train_image_index)

    @property
    def is_attached(self):
        return len(self.relations) > 0

    def __getattr__(self, item):
        if item.startswith("train_"):
            return self.kwargs["train_sample"][item[len("train_"):]]
        elif item.startswith("val_"):
            return self.kwargs["val_sample"][item[len("val_"):]]
        elif item.startswith('metaconcept_') or item.startswith('composite_'):
            return self.kwargs['task'][item]
        else:
            return self.__getattribute__(item)

    def __setattr__(self, item, value):
        if item.startswith("train_"):
            self.kwargs["train_sample"][item[len("train_"):]] = value
        elif item.startswith("val_"):
            self.kwargs["val_sample"][item[len("val_"):]] = value
        elif item.startswith('metaconcept_') or item.startswith('composite_'):
            self.kwargs['task'][item] = value
        else:
            super().__setattr__(item, value)

    @staticmethod
    def sequence2text(tensor, concepts):
        if tensor.ndim == 1:
            return [f"{concepts[tensor[-1]]} from {len(tensor) - 1:01d} masks"]
        else:
            return ['']

    def __mod__(self, dataset):
        return [f"{dataset.named_entries_[self.concept_index]} from {len(self.relations):01d} supports."]

    def evaluate_logits(self, executor, **kwargs):
        # TODO: (@Weiwei) Change this method to adapt to outputs from hierarchical graphical network
        # The actual function that computes logits
        queried_embedding = kwargs["queried_embedding"]
        train_ends, train_query_objects = [], []
        for p in self.train_program:
            q = p.evaluate_token(queried_embedding[self.concept_index])
            out = q(executor)
            e, o = out["end"].max(-1)
            train_ends.append(e)
            train_query_objects.append(o.squeeze(0))
        train_query_objects = torch.stack(train_query_objects)

        val_ends, val_query_objects = [], []
        for p, concept_idx in zip(self.val_program, self.validation_concepts):
            q = p.evaluate_token(queried_embedding[concept_idx])
            out = q(executor)
            val_ends.append(out["end"])
            val_query_objects.append(out["query_object"])
        val_query_objects = torch.stack(val_query_objects)
        output = defaultdict(dict, **kwargs)
        output["train_sample"] = {"end": train_ends, "query_object": train_query_objects}
        output["val_sample"] = {"end": val_ends, "query_object": val_query_objects}
        return output

    def __call__(self, executor):
        # computes queried_embedding and prior_reg
        outputs = executor(self)
        # compute logits from calling MetaLearner.compute_logits(), which just calls the evaluate_logits() method
        # defined above
        outputs = executor.compute_logits(self, **outputs)
        return outputs

    def fewshot(self, training_features, executor):
        # call the gnns
        # Shape: (num_shots = 1, 2*box_dimension)
        self.training_features = training_features
        return executor.network(self)
