import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from models.nn.mlp import MLP


class ConceptExtractionModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        #config = AutoConfig.from_pretrained(cfg.model_name)
        # language encoder
        self.back_bone_model = AutoModel.from_pretrained(cfg.model_name)
        self.model_name = cfg.model_name
        if "t5" in cfg.model_name:
            self.back_bone_model = self.back_bone_model.encoder
            for param in self.back_bone_model.embed_tokens.parameters():
                param.requires_grad = False
        else:
        # freeze the word embedding
            for param in self.back_bone_model.embeddings.word_embeddings.parameters():
                param.requires_grad = False
        # scorer
        # score conditioned on the instruction, node context, and the representation of the concept
        self.mlp_scorer = MLP(
            768,
            cfg.MID_CHANNELS,
            1
        )


    def forward(self, inputs):

        outputs = {}
        batch_size = inputs['contextualized_sentences_tokens'].shape[0]
        device = inputs['contextualized_sentences_tokens'].device
        assert batch_size == 1, "does not deal with batch size > 1!"



        # Shape: (num_choices, seq_len, back_bone_size)
        sentence_representation = self.back_bone_model(inputs['contextualized_sentences_tokens'].squeeze(0), inputs["attention_mask"].squeeze(0)).last_hidden_state

        # Shape: (num_choices, back_bone_size)
        # original_concept_repr = sentence_representation[torch.arange(batch_size), inputs['original_concept_index']]
        # alternative_concept_repr = sentence_representation[torch.arange(batch_size), inputs['alternative_concept_index']]
        sentence_representation = sentence_representation[:, 0, :]
        # Shape: (num_choices)
        logits = self.mlp_scorer(sentence_representation).squeeze(-1)
        outputs["logits"] = logits

        if self.training:
            # Shape: (1)
            target = inputs['label'].squeeze(0)
            loss = F.cross_entropy(logits,target)
            outputs["loss"] = loss
        outputs["probability"] = F.softmax(logits)
        outputs["prediction"] = torch.argmax(logits) 
        return outputs
