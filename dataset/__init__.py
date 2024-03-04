from .clevr import *
from .cub import *
from .gqa import *
from .cub_generalized import *
from .tabletop import *
from .dataset import Dataset
from .node_dataset import NodeClassifierDataset
from .extraction_dataset import ExtractionDataset

__all__ = ["build_dataset", "NodeClassifierDataset", "ExtractionDataset"]


def build_dataset(dataset_cfg, args):
    dataset_fn = Dataset.DATASET_REGISTRY[dataset_cfg.NAME]
    # truncated dataset
    dataset = dataset_fn(dataset_cfg, args)
    if "truncated" in dataset_cfg.OPTS:
        dataset.indices_split = dataset.indices_split[:dataset_cfg.TRUNCATED_SIZE]
    return dataset
