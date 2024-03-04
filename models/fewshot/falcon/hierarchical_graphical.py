import torch
import torch.nn.functional as F

from models.fewshot.falcon.bayes import BayesLearner
from models.fewshot.nn import ConceptGraphGNN, ExampleGraphGNN
from models.nn import Measure
from utils import log_normalize


class HierarchicalGraphicalLearner(BayesLearner):
    def __init__(self, cfg):
        super().__init__(cfg)
        # gnn:
        # input_dim: (n_particles, k, 2 * box_dimension)
        # output_dim: (n_particles, k, 2 * box_dimension)
        # gnn_1: computes a box embedding based on the relation
        self.gnn_1 = ConceptGraphGNN(cfg)
        # gnn_2: computes a box embedding based on the example
        self.gnn_2 = ExampleGraphGNN(cfg)
        # gnn_3: compute the message sent back to the ancestors
        self.gnn_3 = ConceptGraphGNN(cfg)
        self.alpha = cfg.REG.ALPHA
        self.beta = 1.0





    def forward(self, p):
        # p is FewshotHierarchy program
        # Sample the prior embedding
        queried_embeddings = {}
        # Shape: (n_particles, 2 * box_dimension)
        new_concept_embedding = self.prior.sample(self.n_particles)

        # Shape: (n_particles, num_shots, 2*box_dimension)
        features = p.training_features.unsqueeze(0).expand(self.n_particles, -1, -1)

        if p.support_embeddings is not None:
            # Shape: (n_particles, num_support_concepts, 2*box_dimension)
            others = p.support_embeddings.unsqueeze(0).expand(self.n_particles, -1, -1)
            # process the sampled prior with the relation-gnn
            # Shape: (n_particles, num_support_concepts + 1, 2*box_dimension)
            embeddings = torch.cat([others, new_concept_embedding.unsqueeze(1)], 1)

            # Shape: (n_particles, 2*box_dimension)
            new_concept_embedding = self.gnn_1(embeddings, p.relations)[:, -1]
        else:
            # deal with the case where we try to insert the root
            others = None
            embeddings = new_concept_embedding
        # process the concept embedding with the example-gnn
        # Shape: (n_particles, 2, 2*box_dimension)
        embeddings = torch.cat([features, new_concept_embedding.unsqueeze(1)], 1)
        # Shape: (n_particles, 2 * box_dimension)
        new_concept_embedding = self.gnn_2(embeddings)[:, -1]
        # pick the concept embedding that entails the features most
        # Shape: (n_particles)
        log_probs = log_normalize(
            F.logsigmoid(self.entailment(new_concept_embedding.unsqueeze(1), features)).sum(1)) if p.is_fewshot else None
        # Shape: (2*box_dimension)
        new_concept_embedding = self.reduction(new_concept_embedding, log_probs)
        # Shape: (num_support_concepts, 2, 2*box_dimension)
        # The new concept embedding should be detached from the loss that are related to the modules that compute info for parents
        #embeddings = torch.cat([new_concept_embedding.detach().unsqueeze(0).unsqueeze(0).expand(len(p.support_concepts),-1,-1), p.support_embeddings.unsqueeze(1)], 1)
        if p.support_embeddings is not None:
            # Shape: (num_support_concepts, 1, 2*box_dimension)
            features_for_new_concepts = p.training_features.unsqueeze(0).expand(len(p.support_concepts), -1, -1)
            embeddings = torch.cat([features_for_new_concepts,  p.support_embeddings.unsqueeze(1)], 1)
            # Temporal fix: make relations to be the first term
            # (TODO:) Fix the design
            # Shape: (num_support_concepts, 2*box_dimension)
            updated_embeddings = self.gnn_3(embeddings, p.relations[0].unsqueeze(0))[:,-1,:]
            # regularization terms
            # Shape: (num_support_concepts + 1, 2*box_dimension)
            updated_embeddings = torch.cat([updated_embeddings, new_concept_embedding.unsqueeze(0)], 0)
            support_concepts = p.support_concepts.tolist()
        else:
            support_concepts = []
            updated_embeddings = new_concept_embedding.unsqueeze(0)
        for i, concept_index in enumerate([*support_concepts, p.concept_index]):
            queried_embeddings[concept_index] = updated_embeddings[i, :]
        if self.training:
            # This guarantee that the param of prior distribution gets closer to the pretrain box distribution
            # TODO: potential issue with this function call, temporally fixed by the callback function
            # regularization for prior dirichlet distribution for newly inserted concept
            #prior_reg = self.alpha * self.prior_reg(p.concept_embedding.unsqueeze(0))
            # regularization to minimize the shifting of antecedents
            # Using  IoU regularization for now
            # -log P(e_c \intersect e_c')/P(e_c \union e_c')
            # Disable the prior regularization for now because that the current training embedding is random
            #iou_reg =  self.beta * (-self.entailment.measure.iou(p.support_embeddings, updated_embeddings[:-1,:]))
            #return {"queried_embedding": queried_embeddings, "prior_reg": prior_reg}
            return {"queried_embedding": queried_embeddings}
        else:
            return {"queried_embedding": queried_embeddings}

class PrototypeLearner(BayesLearner):
    def __init__(self, cfg):
        super().__init__(cfg)
        # gnn:
        # input_dim: (n_particles, k, 2 * box_dimension)
        # output_dim: (n_particles, k, 2 * box_dimension)
        # gnn_1: computes a box embedding based on the example
        self.gnn_1 = ExampleGraphGNN(cfg)
        # gnn_2: compute the message sent back to the ancestors
        self.gnn_2 = ConceptGraphGNN(cfg)
        self.alpha = cfg.REG.ALPHA
        self.beta = cfg.REG.BETA


    def forward(self, p):
        # p is FewshotHierarchy program
        # Sample the prior embedding
        queried_embeddings = {}
        # Shape: (n_particles, 2 * box_dimension)
        new_concept_embedding = self.prior.sample(self.n_particles)
        # Shape: (n_particles, num_support_concepts, 2*box_dimension)
        others = p.support_embeddings.unsqueeze(0).expand(self.n_particles, -1, -1)
        # Shape: (n_particles, num_shots, 2*box_dimension)
        features = p.training_features.unsqueeze(0).expand(self.n_particles, -1, -1)
        # process the concept embedding with the example-gnn
        # Shape: (n_particles, 2, 2*box_dimension)
        embeddings = torch.cat([features, new_concept_embedding.unsqueeze(1)], 1)
        # Shape: (n_particles, 2 * box_dimension)
        new_concept_embedding = self.gnn_1(embeddings)[:, -1]

        # pick the concept embedding that entails the features most
        # Shape: (n_particles)
        log_probs = log_normalize(
            F.logsigmoid(self.entailment(new_concept_embedding.unsqueeze(1), features)).sum(1)) if p.is_fewshot else None
        # Shape: (2*box_dimension)
        new_concept_embedding = self.reduction(new_concept_embedding, log_probs)

        # Shape: (num_support_concepts, 2, 2*box_dimension)
        # The new concept embedding should be detached from the loss that are related to the modules that compute info for parents
        embeddings = torch.cat([new_concept_embedding.detach().unsqueeze(0).unsqueeze(0).expand(len(p.support_concepts),-1,-1), p.support_embeddings.unsqueeze(1)], 1)
        # Temporal fix: make relations to be the first term
        # (TODO:) Fix the design
        # Shape: (num_support_concepts, 2*box_dimension)
        updated_embeddings = self.gnn_2(embeddings, p.relations[0].unsqueeze(0))[:,-1,:]

        # regularization terms
        # Shape: (num_support_concepts + 1, 2*box_dimension)
        updated_embeddings = torch.cat([updated_embeddings, new_concept_embedding.unsqueeze(0)], 0)

        for i, concept_index in enumerate([*p.support_concepts.tolist(), p.concept_index]):
            queried_embeddings[concept_index] = updated_embeddings[i, :]
        if self.training:
            # This guarantee that the param of prior distribution gets closer to the pretrain box distribution
            # TODO: potential issue with this function call, temporally fixed by the callback function
            # regularization for prior dirichlet distribution for newly inserted concept
            #prior_reg = self.alpha * self.prior_reg(p.concept_embedding.unsqueeze(0))
            # regularization to minimize the shifting of antecedents
            # Using  IoU regularization for now
            # -log P(e_c \intersect e_c')/P(e_c \union e_c')
            # Disable the prior regularization for now because that the current training embedding is random
            #iou_reg =  self.beta * (-self.entailment.measure.iou(p.support_embeddings, updated_embeddings[:-1,:]))
            #return {"queried_embedding": queried_embeddings, "prior_reg": prior_reg}
            #return {"queried_embedding": queried_embeddings, "iou_reg": iou_reg}
            return {"queried_embedding": queried_embeddings}
        else:
            return {"queried_embedding": queried_embeddings}

class FalconGraphicalLearner(BayesLearner):
    # This is just porting FALCON's model for a thorough comparison
    def __init__(self, cfg):
        super().__init__(cfg)
        # gnn:
        # input_dim: (n_particles, k, 2 * box_dimension)
        # output_dim: (n_particles, k, 2 * box_dimension)
        self.gnn_1 = ConceptGraphGNN(cfg)
        self.gnn_2 = ExampleGraphGNN(cfg)
    def forward(self, p):
        # p is FewshotHierarchy program
        # Sample the prior embedding
        queried_embeddings = {}
        # Shape: (n_particles, 2 * box_dimension)
        new_concept_embedding = self.prior.sample(self.n_particles)
        if p.support_embeddings is not None:
            # Shape: (n_particles, num_support_concepts, 2*box_dimension)
            others = p.support_embeddings.unsqueeze(0).expand(self.n_particles, -1, -1)

            # process the sampled prior with the relation-gnn
            # Shape: (n_particles, num_support_concepts + 1, 2*box_dimension)
            embeddings = torch.cat([others, new_concept_embedding.unsqueeze(1)], 1)
            # Shape: (n_particles, num_support_concepts + 1, 2*box_dimension)
            new_concept_embedding = self.gnn_1(embeddings, p.relations)[:, -1]

        # Shape: (n_particles, num_shots, 2*box_dimension)
        features = p.training_features.unsqueeze(0).expand(self.n_particles, -1, -1)
        # process the concept embedding with the example-gnn
        # Shape: (n_particles, num_support_concepts + 1, 2*box_dimension)
        embeddings = torch.cat([features, new_concept_embedding.unsqueeze(1)], 1)
        # Shape: (n_particles, 2 * box_dimension)
        new_concept_embedding = self.gnn_2(embeddings)[:, -1]


        # Shape: (n_particles)
        log_probs = log_normalize(
            F.logsigmoid(self.entailment(new_concept_embedding.unsqueeze(1), features)).sum(
                1)) if p.is_fewshot else None
        new_concept_embedding = self.reduction(new_concept_embedding, log_probs)
        # prior_reg is for computing the loss for prior distribution matching
        # hope to match Dirichlet distribution
        if p.support_embeddings is not None:
            support_concepts = p.support_concepts.tolist()
        else:
            support_concepts = []
        for i, concept_index in enumerate(support_concepts):
            queried_embeddings[concept_index] = p.support_embeddings[i]
        queried_embeddings[p.concept_index] = new_concept_embedding
        if self.training:
            # not sure why this is not taken from the current embedding, but following source code
            #prior_reg = self.prior_reg(p.concept_embedding.unsqueeze(0))
            return {"queried_embedding": queried_embeddings}
        else:
            return {"queried_embedding": queried_embeddings}