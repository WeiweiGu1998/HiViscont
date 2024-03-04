import os
import torch
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from experiments.defaults import C
from dataset.utils import DataLoader, cycle
from dataset import NodeClassifierDataset
from models import NodeClassifierModel
from solver import make_scheduler, Optimizer
from utils import to_cuda
from utils import setup_logger, start_up, ArgumentParser, FLAGS, data_parallel
from utils import Checkpointer, mkdir
from yacs.config import CfgNode as CN
import copy
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_name', default='bert-base-cased', help='backbone transformer model')
    parser.add_argument('-o', '--output_dir', default='node_model', help='output directory')
    args = parser.parse_args()
    return args

def get_test_config(model_name):
    M = CN()
    M.model_name = model_name
    M.MID_CHANNELS = 200
    DS = CN()
    DS.model_name = model_name
    DS.train = "train"
    DS.val = "val"
    DS.split = "train"
    return M, DS

def train(cfg):
    logger = logging.getLogger("falcon_logger")
    train_set = NodeClassifierDataset(cfg.DS)
    val_ds = copy.deepcopy(cfg.DS)
    val_ds.split = "val"
    val_set = NodeClassifierDataset(val_ds)
    train_loader = DataLoader(train_set, cfg)

    model = NodeClassifierModel(cfg.M).to("cuda")
    # add the special tokens also to the pretrained bert model
    # this need to be done when loading the model everytime.
    model.back_bone_model.resize_token_embeddings(len(train_set.tokenizer))
    param_groups = [{
        'params': [p for n, p in model.named_parameters()],
        'lr': 3e-5, 'weight_decay': 1e-5
    }]
    optimizer = AdamW(params=param_groups)
    schedule = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(len(train_set)//cfg.SOLVER.BATCH_SIZE)* 5,
        num_training_steps=(len(train_set)//cfg.SOLVER.BATCH_SIZE) * 50
    )
    #checkpointer = Checkpointer(cfg, model, optimizer, scheduler)
    #start_iteration = checkpointer.load(args.iteration)
    state = dict(
        model=model.state_dict()
    )
    iteration = 0
    max_iter = cfg.SOLVER.MAX_ITER
    validation_period = cfg.SOLVER.VALIDATION_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    logger.info("Start training.")

    num_iter = 10
    best_epoch = -1
    train_accr_of_best_epoch = -1
    best_val_accr = -1

    for i in range(num_iter):
        # train
        logger.info(f"Epoch: {i}")
        model.train()
        train_targets = []
        train_predictions = []

        for inputs in train_loader:
            iteration = iteration + 1
            inputs = to_cuda(inputs)
            outputs = model(inputs)
            train_targets.append(inputs["label"])
            train_predictions.append(outputs["prediction"])
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_NORM)
            optimizer.step()
            schedule.step()
        train_targets = torch.cat(train_targets, dim=0)
        train_predictions = torch.cat(train_predictions, dim=0)
        train_accuracy = torch.sum(train_targets == train_predictions) / len(train_targets)
        logger.info(f"Train Accuracy: {train_accuracy}")
        # validation
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            model.eval()
            val_loader = DataLoader(val_set, cfg)
            for inputs in val_loader:
                iteration = iteration + 1
                inputs = to_cuda(inputs)
                outputs = model(inputs)
                val_targets.append(inputs["label"])
                val_predictions.append(outputs["prediction"])
            val_targets = torch.cat(val_targets, dim=0)
            val_predictions = torch.cat(val_predictions, dim=0)
            val_accuracy = torch.sum(val_targets == val_predictions) / len(val_targets)

        logger.info(f"Validation Accuracy: {val_accuracy}")
        if val_accuracy > best_val_accr:
            best_epoch = i
            best_val_accr = val_accuracy
            train_accr_of_best_epoch = train_accuracy
            torch.save(state, os.path.join(cfg.OUTPUT_DIR, "best_model.mdl"))
        elif val_accuracy == best_val_accr:
            if train_accuracy > train_accr_of_best_epoch:
                best_epoch = i
                train_accr_of_best_epoch = train_accuracy
                torch.save(state, os.path.join(cfg.OUTPUT_DIR, "best_model.mdl"))


    logger.info("training done")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best Validation Accuracy: {best_val_accr}")
    logger.info(f"Training Accuracy of best epoch: {train_accr_of_best_epoch}")

    # # test for loading model
    # model_path = os.path.join(cfg.OUTPUT_DIR, "best_model.mdl")
    # temp_state = torch.load(model_path)
    # new_model = NodeClassifierModel(cfg.M)
    # new_model.back_bone_model.resize_token_embeddings(len(train_set.tokenizer))
    # new_model.load_state_dict(temp_state["model"])
    # new_model = new_model.to("cuda")
    # val_targets = []
    # val_predictions = []
    # with torch.no_grad():
    #     new_model.eval()
    #     val_loader = DataLoader(val_set, cfg)
    #     for inputs in val_loader:
    #         iteration = iteration + 1
    #         inputs = to_cuda(inputs)
    #         outputs = new_model(inputs)
    #         val_targets.append(inputs["target_index"])
    #         val_predictions.append(outputs["prediction"])
    #     val_targets = torch.cat(val_targets, dim=0)
    #     val_predictions = torch.cat(val_predictions, dim=0)
    #     val_accuracy = torch.sum(val_targets == val_predictions) / len(val_targets)
    #
    # logger.info(f"Validation Accuracy: {val_accuracy}")


def main():
    args = parse_args()
    M, DS = get_test_config(args.model_name)
    cfg = copy.deepcopy(C)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, args.output_dir)
    output_dir = mkdir(cfg.OUTPUT_DIR)
    logger = setup_logger("falcon_logger", os.path.join(output_dir, "train_log.txt"))
    cfg.M = M
    cfg.DS = DS
    cfg.SOLVER.BATCH_SIZE = 32
    start_up()
    train(cfg)


if __name__ == "__main__":
    main()
