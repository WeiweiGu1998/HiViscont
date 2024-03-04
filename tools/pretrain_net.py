import datetime
import logging
import os
import shutil
import time

import psutil
import torch
import random
import numpy as np
from dataset.utils import DataLoader, cycle
from experiments import cfg_from_args
from models import build_model
from solver import make_scheduler, Optimizer
from tools.dataset_catalog import DatasetCatalog
from utils import Checkpointer, mkdir
from utils import setup_logger, SummaryWriter, Metric
from utils import start_up, ArgumentParser, FLAGS, data_parallel
from utils import to_cuda, gather_loss
from visualization import build_visualizer


def train(cfg, args):
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    logger = logging.getLogger("falcon_logger")
    logger.info("Setting up datasets.")
    train_set = DatasetCatalog(cfg).get(cfg.DATASETS.TRAIN, cfg.DATASETS.DOMAIN, args)
    val_set = DatasetCatalog(cfg).get(cfg.DATASETS.VAL, cfg.DATASETS.DOMAIN, args)
    train_loader = DataLoader(train_set, cfg)
    logger.info("Setting up models.")
    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    model = build_model(cfg).to("cuda")
    model = data_parallel(model, gpu_ids)

    logger.info("Setting up utilities.")
    output_dir = cfg.OUTPUT_DIR

    optimizer = Optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    checkpointer = Checkpointer(cfg, model, optimizer, scheduler)
    start_iteration = checkpointer.load(args.iteration)

    max_iter = cfg.SOLVER.MAX_ITER
    validation_period = cfg.SOLVER.VALIDATION_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    max_val_iter = cfg.SOLVER.VALIDATION_LIMIT
    summary_writer = SummaryWriter(output_dir)
    #train_visualizer = build_visualizer(cfg.VISUALIZATION, train_set, summary_writer)
    #val_visualizer = build_visualizer(cfg.VISUALIZATION, val_set, summary_writer)
    summary_writer.add_text("cfg", str(cfg).replace("\n", "\n    "))
    train_metrics = Metric(delimiter="  ", summary_writer=summary_writer)

    logger.info("Start training.")
    model.train()
    train_set.callback(start_iteration)
    start_training_time = time.time()
    last_batch_time = time.time()
    # store box registry for the model for dynamic validation
    if cfg.RESET_BOX:
        model.reset_box(cfg.MODEL)
        model = model.to("cuda")

    for iteration, inputs in enumerate(cycle(train_loader), start_iteration):
        data_time = time.time() - last_batch_time
        iteration = iteration + 1
        inputs = to_cuda(inputs)
        outputs = model(inputs)
        loss_dict = gather_loss(outputs)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_NORM)
        optimizer.step()
        scheduler.step_iteration()
        model.callback(inputs, outputs)
        train_set.callback(iteration)

        batch_time = time.time() - last_batch_time
        last_batch_time = time.time()
        train_metrics.update(**loss_dict)
        train_metrics.update(batch_time=batch_time, data_time=data_time, lr=optimizer.lr)

        if iteration % 20 == 0:
            eta = datetime.timedelta(seconds=int(train_metrics.batch_time.mean * (max_iter - iteration)))
            memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            ram = psutil.virtual_memory().percent
            logger.info(train_metrics.delimiter.join(
                [f"eta: {eta}", f"iter: {iteration}", f"{str(train_metrics)}", f"max mem: {memory:.0f}",
                    f"ram: {ram}"]))
            train_metrics.log_summary(train_set.tag, iteration)

        #if iteration % 100 == 0:
        #    train_visualizer.visualize(inputs, outputs, model, iteration)

        if iteration % checkpoint_period == 0:
            checkpointer.save(iteration)

        if iteration >= max_iter:
            break
        del inputs, outputs

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f"Total training time: {total_time_str} ({total_training_time / max_iter:.4f} s / it)")


def main():
    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    if args.restart:
        restart = input("Type 1 if restart\n")
        if "1" in restart and os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR, ignore_errors=True)
        elif "2" in restart:
            FLAGS["SKIP_CACHE"] = True

    output_dir = mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("falcon_logger", os.path.join(output_dir, "train_log.txt"))
    start_up()

    logger.info(f"Running with args:\n{args}")
    logger.info(f"Running with config:\n{cfg}")
    train(cfg, args)


if __name__ == "__main__":
    main()