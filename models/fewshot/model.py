from collections import defaultdict

import torch
from torch import nn

from models.fewshot.nn import FewshotLoss
from models.fewshot.program_executor import FewshotProgramExecutor, FewshotHierarchyProgramExecutor
from models.nn import build_box_registry, CachedFeatureExtractor
from models.pretrain.program_executor import PretrainProgramExecutor
from utils import check_entries, map_wrap
from utils import collate_fn, freeze, compose_image

import copy

class FewshotModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.box_registry = build_box_registry(cfg)
        freeze(self.box_registry)
        self.fixed_box_registry = copy.deepcopy(self.box_registry)
        freeze(self.fixed_box_registry)

        self.feature_extractor = CachedFeatureExtractor(cfg)
        self.program_executor = FewshotProgramExecutor(cfg)
        self.loss = FewshotLoss(cfg)

    def reset_box(self, cfg):
        self.box_registry = build_box_registry(cfg)
        freeze(self.box_registry)
        self.fixed_box_registry = copy.deepcopy(self.box_registry)
        freeze(self.fixed_box_registry)

    @property
    def concept_entries(self):
        return len(self.box_registry)

    @map_wrap
    @map_wrap
    def _forward_pretrained(self, index, pf):
        return self.feature_extractor.get(index, pf)

    @map_wrap
    @map_wrap
    def _forward_masked(self, index, image, mask):
        return self.feature_extractor.get(index, compose_image(image, mask))

    @map_wrap
    @map_wrap
    def _forward_unmasked(self, index, image):
        return self.feature_extractor.get(index, image.unsqueeze(0))

    def forward_feature_extractor(self, samples):
        if "pretrained" in samples:
            features, relations = self._forward_pretrained(*map(samples.get, ["image_index", "pretrained"]))
        elif "mask" in samples:
            features, relations = self._forward_masked(*map(samples.get, ["image_index", "image", "mask"]))
        else:
            features, relations = self._forward_unmasked(*map(samples.get, ["image_index", "image"]))
        return {"features": features, "relations": relations}

    def make_kwargs(self, inputs, train_samples, val_samples, i):
        kwargs = defaultdict(dict, device=self.box_registry.device)
        for group, samples in zip(["train_sample", "val_sample"], [train_samples, val_samples]):
            for k, v in inputs[group].items():
                kwargs[group][k] = v[i]
            kwargs[group]["features"] = samples["features"][i]
            kwargs[group]["relations"] = samples["relations"][i]
        for k, v in inputs['task'].items():
            kwargs['task'][k] = v[i]
        return kwargs

    def forward(self, inputs):
        check_entries(self.concept_entries, inputs["info"]["concept_entries"][0])
        train_samples = self.forward_feature_extractor(inputs["train_sample"])
        val_samples = self.forward_feature_extractor(inputs["val_sample"])
        outputs = defaultdict(list)
        for i, p in enumerate(inputs["program"]):
            kwargs = self.make_kwargs(inputs, train_samples, val_samples, i)
            q = p.evaluate(self.box_registry, **kwargs)
            # This line actually computes the queried embedding from the training sample
            o = self.program_executor(q)
            for k, v in o.items():
                outputs[k].append(v)
            outputs["program"].append(q)
        # The output dict contains:
        #   queried_embedding
        #   program
        #   prior_reg (if training)
        outputs = collate_fn(outputs)
        losses = {}
        if self.training:
            losses = self.loss(outputs, inputs)

        outputs["train_sample"].update(train_samples)
        outputs["val_sample"].update(val_samples)
        return {**outputs, **losses}

    def callback(self, inputs, outputs):
        if not self.training and inputs["info"]["split"][0] != "train" and not inputs['info']['use_text'][0]:
            # Fill the resulting model with fewshot results
            with torch.no_grad():
                for p, e in zip(inputs["program"], outputs["queried_embedding"]):
                    self.box_registry[p.concept_index] = e
        # Fill in the cache for extracted features
        if not self.feature_extractor.has_cache:
            for group in ["train_sample", "val_sample"]:
                for features, relations, indices in zip(outputs[group]["features"], outputs[group]["relations"],
                        inputs[group]["image_index"]):
                    for feature, relation, index in zip(features, relations, indices):
                        self.feature_extractor.set(index, (feature, relation))

    @property
    def rep(self):
        return self.feature_extractor.rep


class FewshotHierarchyModel(FewshotModel):

    def __init__(self, cfg):
        # for now, use the same feature extractor, loss function, and box registry
        # as fewshot model
        super().__init__(cfg)
        # This should be a different feature extractor
        self.program_executor = FewshotHierarchyProgramExecutor(cfg)
        self.pretrain_program_executor = PretrainProgramExecutor(cfg)

    def make_kwargs(self, inputs, train_samples, val_samples, i):
        kwargs = defaultdict(dict, device=self.box_registry.device)
        for group, samples in zip(["train_sample", "val_sample"], [train_samples, val_samples]):
            for k, v in inputs[group].items():
                kwargs[group][k] = v[i]
            kwargs[group]["features"] = samples["features"][i]
            kwargs[group]["relations"] = samples["relations"][i]
        for k, v in inputs['task'].items():
            kwargs['task'][k] = v[i]
        # also need the concept index
        kwargs['concept_index'] = inputs['concept_index'][i]
        return kwargs

    def forward(self, inputs):
        check_entries(self.concept_entries, inputs["info"]["concept_entries"][0])
        train_samples = self.forward_feature_extractor(inputs["train_sample"])
        val_samples = self.forward_feature_extractor(inputs["val_sample"])
        outputs = defaultdict(list)
        for i, p in enumerate(inputs["program"]):
            # p is a Composite program where p.gt_embeddings is the index of related concepts
            kwargs = self.make_kwargs(inputs, train_samples, val_samples, i)
            p = p.to_fewshot_hierarchy()
            p.to_evaluate(self.box_registry, **kwargs)
            # This line actually computes the queried embedding from the training sample
            o = self.program_executor(p)
            for k, v in o.items():
                outputs[k].append(v)
            outputs["program"].append(p)
        outputs = collate_fn(outputs)
        losses = {}
        if self.training:
            losses = self.loss(outputs, inputs)
        outputs["train_sample"].update(train_samples)
        outputs["val_sample"].update(val_samples)
        return {**outputs, **losses}


    def inference_fewshot_inputs(self, inputs):
        check_entries(self.concept_entries, inputs["info"]["concept_entries"][0])
        train_samples = self.forward_feature_extractor(inputs["train_sample"])
        val_samples = self.forward_feature_extractor(inputs["val_sample"])
        outputs = defaultdict(list)
        for i, p in enumerate(inputs["program"]):
            # p is a Composite program where p.gt_embeddings is the index of related concepts
            kwargs = self.make_kwargs(inputs, train_samples, val_samples, i)
            p = p.to_fewshot_hierarchy()
            p.to_evaluate(self.box_registry, **kwargs)
            # This line actually computes the queried embedding from the training sample
            o = self.program_executor(p)
            for k, v in o.items():
                outputs[k].append(v)
            outputs["program"].append(p)
        outputs = collate_fn(outputs)
        outputs["train_sample"].update(train_samples)
        outputs["val_sample"].update(val_samples)
        return outputs


    def inference_pretrain_inputs(self, inputs):
        # this method combine the pretrain network
        check_entries(self.concept_entries, inputs["info"]["concept_entries"][0])
        features = self.forward_feature_extractor(inputs)
        relations = features['relations'][0]
        features = features['features'][0]
        program = inputs["program"]
        outputs = defaultdict(list)

        for p, feature, relation in zip(program, features, relations):
            q = p.evaluate(self.box_registry, features=feature, relations=relation)
            output = self.pretrain_program_executor(q)
            for k, v in output.items():
                outputs[k].append(v)
            outputs["program"].append(q)
        outputs = collate_fn(outputs, ("end",))
        return outputs

    def inference(self, inputs):
        if 'train_sample' not in inputs.keys():
            inputs['mask'] = inputs['mask'].unsqueeze(0)
            inputs['image'] = inputs['image'].unsqueeze(0)
            inputs['image_index'] = [inputs['image_index']]
            return self.inference_pretrain_inputs(inputs)
        else:
            return self.inference_fewshot_inputs(inputs)

    def callback(self, inputs, outputs):
        #if not self.training and inputs["info"]["split"][0] != "train" and not inputs['info']['use_text'][0]:
            # Fill the resulting model with fewshot results

        # This update only the concepts that are queried
        with torch.no_grad():
            for p, updated_concepts in zip(inputs["program"], inputs["concept_index"]):
                # update all concept index
                for c in updated_concepts:
                    #if inputs["info"]["split"][0] == "val" and c != p.concept_index:
                    #if c != p.concept_index:
                        # for validation, we update only the validation concept
                        # TODO: (@Weiwei) This is just a temporary fix
                        #continue
                    self.box_registry[c] = outputs["queried_embedding"][c]
        # Fill in the cache for extracted features
        if not self.feature_extractor.has_cache:
            for group in ["train_sample", "val_sample"]:
                for features, relations, indices in zip(outputs[group]["features"], outputs[group]["relations"],
                        inputs[group]["image_index"]):
                    for feature, relation, index in zip(features, relations, indices):
                        self.feature_extractor.set(index, (feature, relation))
        freeze(self.box_registry)


    def update_concept_embedding(self, queried_embedding):
        with torch.no_grad():
            for concept_index, embedding in queried_embedding.items():
                self.box_registry[concept_index] = queried_embedding[concept_index]
        freeze(self.box_registry)

    def set_fixed_box_registry(self):
        self.fixed_box_registry = copy.deepcopy(self.box_registry)
        freeze(self.fixed_box_registry)
    def reset_box_registry(self):
        self.box_registry = copy.deepcopy(self.fixed_box_registry)
        freeze(self.box_registry)
