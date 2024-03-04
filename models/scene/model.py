import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from models.nn.mlp import MLP


class NodeClassifierModel(nn.Module):

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
        batch_size = inputs['contextualized_query_tokens'].shape[0]
        device = inputs['contextualized_query_tokens'].device



        # Shape: (batch_size, seq_len, back_bone_size)
        sentence_representation = self.back_bone_model(inputs['contextualized_query_tokens'].squeeze(1), inputs["contextualized_query_attention_mask"].squeeze(1)).last_hidden_state


        # Shape: (batch_size, back_bone_size)
        # original_concept_repr = sentence_representation[torch.arange(batch_size), inputs['original_concept_index']]
        # alternative_concept_repr = sentence_representation[torch.arange(batch_size), inputs['alternative_concept_index']]
        sentence_representation = sentence_representation[:, 0, :]
        # Shape: (batch_size)
        logits = self.mlp_scorer(sentence_representation).squeeze(-1)
        outputs["logits"] = logits

        if self.training:
            # Shape: (batch_size)
            target = inputs['label'].float()
            loss = F.binary_cross_entropy_with_logits(logits,target)
            outputs["loss"] = loss
        outputs["probability"] = F.sigmoid(logits)
        outputs["prediction"] = (logits >= 0)

        # Shape: (batch_size, 2, back_bone_size)
        #sentence_representation = sentence_representation[:, 0, :].unsqueeze(1).expand(-1, 2, -1)
        #original_concept_representation = sentence_representation[:, -4, :]
        #alternative_concept_representation = sentence_representation[:, -2, :]

        # Shape: (batch_size, back_bone_size)

        # Shape: (batch_size,1)
        # original_concept_score = self.cos_similarity_scorer(sentence_representation,
        #                                                     original_concept_repr).unsqueeze(1)
        # alternative_concept_score = self.cos_similarity_scorer(sentence_representation,
        #                                                        alternative_concept_repr).unsqueeze(1)
        # Shape: (batch_size, 1, back_bone_size)
        #original_concept_representation = self.back_bone_model.embeddings.word_embeddings(inputs['original_concept_tokens'].squeeze(1))
        #alternative_concept_representation = self.back_bone_model.embeddings.word_embeddings(inputs['alternative_concept_tokens'].squeeze(1))
        # Shape: (batch_size, back_bone_size)
        #if "t5" in self.model_name:
        #    original_concept_representation = self.back_bone_model.embed_tokens(inputs['original_concept_tokens'].squeeze(1)).squeeze(1)
        #    alternative_concept_representation = self.back_bone_model.embed_tokens(inputs['alternative_concept_tokens'].squeeze(1)).squeeze(1)
        #else:
        #    original_concept_representation = self.back_bone_model.embeddings.word_embeddings(inputs['original_concept_tokens'].squeeze(1)).squeeze(1)
        #    alternative_concept_representation = self.back_bone_model.embeddings.word_embeddings(inputs['alternative_concept_tokens'].squeeze(1)).squeeze(1)

        # Shape: (batch_size, 2, embedding_size)
        #concept_representation = torch.cat([original_concept_representation, alternative_concept_representation], dim=1)
        # Shape: (batch_size, 2, 2 * back_bone_size)
        #representation = torch.cat([sentence_representation, concept_representation], 2)

        # Shape: (batch_size, 2)
        # raw score for each concept
        #logits = self.mlp_scorer(representation).squeeze(-1)
        # Shape: (batch_size,1)
        #original_concept_score = self.cos_similarity_scorer(sentence_representation, original_concept_representation).unsqueeze(1)
        #alternative_concept_score = self.cos_similarity_scorer(sentence_representation, alternative_concept_representation).unsqueeze(1)
        # logits = torch.cat([original_concept_score, alternative_concept_score], dim=1)
        # outputs["logits"] = logits
        #
        # if self.training:
        #     # Shape: (batch_size, 1)
        #     target = inputs['target_index']
        #     # compute cross entropy loss
        #     loss = F.cross_entropy(logits, target)
        #     outputs["loss"] = loss
        # # Shape: (batch_size, 2)
        # outputs["probability"] = F.softmax(logits, 1)
        # # Shape: (batch_size)
        # outputs["prediction"] = torch.argmax(logits, dim=1)
        return outputs
