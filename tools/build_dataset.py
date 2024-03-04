from dataset.cub_generalized.cub_generalized_concept_dataset import *
from dataset.cub_generalized.cub_generalized_fewshot_dataset import *
from dataset.tabletop.tabletop_concept_dataset import *
from dataset.tabletop.tabletop_fewshot_dataset import *
from dataset.cub.cub_concept_dataset import *
from dataset.cub.cub_fewshot_dataset import *
from yacs.config import CfgNode as CN
from utils import join
from utils import ArgumentParser

import numpy as np
import torch


def build_test(CUB_ROOT, split_seed):
    dataset_cfg = CN(
        dict(
            NAME="cub_concept_full_builder",
            SPLIT="all",
            DOMAIN="",
            ROOT=CUB_ROOT,
            OPTS=[],
            SHOT_K=1,
            SPLIT_SEED=split_seed,
            QUERY_K=30,
            DROPOUT_RATE=.2,
            HAS_MASK=True,
            TRUNCATED_SIZE=50,
            USE_TEXT=False,
            SPLIT_RATIO=[.7, .0, .3],
        )
    )
    builder_dataset = CubConceptFullBuilderDataset(dataset_cfg, args)
    builder_dataset.dump_questions()

def build_pretrain(CUB_ROOT, split_seed):
    dataset_cfg = CN(
        dict(
            NAME="cub_concept_support_builder",
            SPLIT="all",
            DOMAIN="",
            ROOT=CUB_ROOT,
            OPTS=[],
            SHOT_K=1,
            SPLIT_SEED=split_seed,
            QUERY_K=30,
            DROPOUT_RATE=.2,
            HAS_MASK=True,
            TRUNCATED_SIZE=50,
            USE_TEXT=False,
            SPLIT_RATIO=[.7, .0, .3],
        )
    )
    builder_dataset = CubConceptSupportBuilderDataset(dataset_cfg, args)
    builder_dataset.dump_questions()


def build_warmup(CUB_ROOT, split_seed):
    dataset_cfg = CN(
        dict(
            NAME="cub_concept_warmup_support_builder",
            SPLIT="all",
            DOMAIN="",
            ROOT=CUB_ROOT,
            OPTS=[],
            SHOT_K=1,
            SPLIT_SEED=split_seed,
            QUERY_K=30,
            DROPOUT_RATE=.2,
            HAS_MASK=True,
            TRUNCATED_SIZE=50,
            USE_TEXT=False,
            SPLIT_RATIO=[.7, .0, .3],
        )
    )
    builder_dataset = CubConceptWarmupSupportBuilderDataset(dataset_cfg, args)
    builder_dataset.dump_questions()


def build_fewshot(CUB_ROOT, split_seed):
    dataset_cfg = CN(
        dict(
            NAME="cub_fewshot_hierarchy_builder",
            SPLIT="all",
            ROOT=CUB_ROOT,
            DOMAIN="",
            OPTS=[],
            SHOT_K=1,
            SPLIT_SEED=split_seed,
            QUERY_K=30,
            DROPOUT_RATE=.2,
            HAS_MASK=True,
            TRUNCATED_SIZE=50,
            USE_TEXT=False,
            SPLIT_RATIO=[.7, .0, .3],
        )
    )
    builder_dataset = CubFewshotHierarchyBuilderDataset(dataset_cfg, args)
    builder_dataset.dump_questions()

def build(args):
    DATASET_ROOT = "/home/local/ASUAD/weiweigu/data"
    CUB_ROOT = join(DATASET_ROOT, "CUB-200-2011")
    split_seed = 104
    random.seed(split_seed)
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)
    build_warmup(CUB_ROOT, split_seed)
    build_pretrain(CUB_ROOT, split_seed)
    build_fewshot(CUB_ROOT, split_seed)
    build_test(CUB_ROOT, split_seed)




if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    build(args)

