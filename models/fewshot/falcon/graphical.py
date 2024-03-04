import torch
import torch.nn.functional as F

from models.fewshot.falcon.bayes import BayesLearner
from models.fewshot.nn import ConceptGraphGNN, ExampleGraphGNN
from utils import log_normalize


class GraphicalLearner(BayesLearner):
    def __init__(self, cfg):
        super().__init__(cfg)
        # gnn:
        # input_dim: (n_particles, k, 2 * box_dimension)
        # output_dim: (n_particles, k, 2 * box_dimension)
        self.gnn_1 = ConceptGraphGNN(cfg)
        self.gnn_2 = ExampleGraphGNN(cfg)

    def forward(self, p):
        # Shape: (n_particles, 2 * box_dimension)
        queried = self.prior.sample(self.n_particles)
        # Shape: (n_particles, num_support_concepts, 2*box_dimension)
        others = p.support_embeddings.unsqueeze(0).expand(self.n_particles, -1, -1)
        # Shape: (n_particles, num_shots, 2*box_dimension)
        features = p.train_features.unsqueeze(0).expand(self.n_particles, -1, -1)

        # process the sampled prior with the relation-gnn
        if p.is_attached:
            # Shape: (n_particles, num_support_concepts + 1, 2*box_dimension)
            embeddings = torch.cat([others, queried.unsqueeze(1)], 1)
            # Shape: (n_particles, 2 * box_dimension)
            queried = self.gnn_1(embeddings, p.relations)[:, -1]

        # process the concept embedding with the example-gnn
        if p.is_fewshot:
            # Shape: (n_particles, num_support_concepts + 1, 2*box_dimension)
            embeddings = torch.cat([features, queried.unsqueeze(1)], 1)
            # Shape: (n_particles, 2 * box_dimension)
            queried = self.gnn_2(embeddings)[:, -1]

        # pick the concept embedding that entails the features most
        # Shape: (n_particles)
        log_probs = log_normalize(
            F.logsigmoid(self.entailment(queried.unsqueeze(1), features)).sum(1)) if p.is_fewshot else None
        # Shape: (2*box_dimension)
        queried_embedding = self.reduction(queried, log_probs)

        if self.training:
            # regularization for prior dirichlet distribution for newly inserted concept
            prior_reg = self.prior_reg(p.parent.concept_index.unsqueeze(0))
            return {"queried_embedding": queried_embedding, "prior_reg": prior_reg}
        else:
            return {"queried_embedding": queried_embedding}
