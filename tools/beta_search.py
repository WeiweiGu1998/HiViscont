import optuna
import json
import random
import torch
import numpy
import copy
import os
import json
import sys
import glob

from experiments import cfg_from_args
from tools.train_hierarchy_net import train
from utils import start_up, ArgumentParser, FLAGS, data_parallel
from utils import Checkpointer, mkdir, setup_logger

def main():
    parser = ArgumentParser()
    args = parser.parse_args()
    root_cfg = cfg_from_args(args)
    root_cfg.defrost()
    beta_range = [-1] + list(range(20))
    for i in beta_range:
        cfg = copy.deepcopy(root_cfg)
        if i < 0:
            cfg.MODEL.REG.BETA = 0.0
            cfg.OUTPUT_DIR = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/output/prototype_beta_search/0"
        else:
            cfg.MODEL.REG.BETA = 2 ** i
            cfg.OUTPUT_DIR = f"/home/local/ASUAD/weiweigu/research/FALCON-Generalized/output/prototype_beta_search/{2 ** i}"
        cfg.freeze()
        output_dir = mkdir(cfg.OUTPUT_DIR)
        logger = setup_logger("falcon_logger", os.path.join(output_dir, "train_log.txt"))
        start_up()

        logger.info(f"Running with args:\n{args}")
        logger.info(f"Running with config:\n{cfg}")
        train(cfg, args)
        logger.handlers[0].close()
        logger.removeHandler(logger.handlers[0])
        logger.handlers[0].close()
        logger.removeHandler(logger.handlers[0])



if __name__ == "__main__":
    main()