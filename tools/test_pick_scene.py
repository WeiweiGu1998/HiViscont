import datetime
import logging
import os
import time
import random
from PIL import Image

import torch
import numpy as np

from dataset.utils import DataLoader
from experiments import cfg_from_args
from models import build_model
from models.programs import build_program, to_batch

from tools.dataset_catalog import DatasetCatalog
from utils import Checkpointer, mkdir
from utils import setup_logger, SummaryWriter, Metric
from utils import start_up, ArgumentParser, data_parallel
from utils import to_cuda, collate_fn
from visualization import build_visualizer

from collections import defaultdict
import copy
import json

import torchvision.transforms.functional as TF

from dataset.dataset import Dataset, BuilderDataset
from dataset.utils import FixedResizeTransform, ProgramVocab
from dataset.utils import sample_with_ratio, WordVocab
from utils import join, load, read_image, file_cached, mask2bbox, nonzero
SCENE_ROOT = "/home/local/ASUAD/weiweigu/data/test_dataset/House/pick_scenes"


def load_image_from_path(image_path):
    t = TF.to_tensor(read_image(image_path))
    return t

def get_mask_and_label_for_image_with_annotation_path(annotation_path, image_tensor):
    with open(annotation_path) as f:
        annotation = json.load(f)
    if 'label' in annotation.keys():
        label = annotation['label']
    else:
        label = 0
    objects = annotation['shapes']
    points = [obj['points'] for obj in objects]
    x_s = [[x[0] for x in obj_pt] for obj_pt in points]
    y_s = [[x[1] for x in obj_pt] for obj_pt in points]
    bounding_boxes = [[[min(x_s[i]), max(x_s[i])], [min(y_s[i]), max(y_s[i])]] for i in range(len(x_s))]
    image = image_tensor
    mask = torch.zeros(size=[len(objects),image.shape[1], image.shape[2]])
    for h in range(mask.shape[1]):
        for w in range(mask.shape[2]):
            for i in range(len(objects)):
                if (w >= bounding_boxes[i][0][0]) and (w <= bounding_boxes[i][0][1]) and (h >= bounding_boxes[i][1][0]) and (h <= bounding_boxes[i][1][1]):
                    mask[i, h, w] = 1
    mask = (mask > 0)
    concept = objects[label]['label']
    return mask, label, concept

def build_question_program(concepts):
    assert len(concepts) <= 2, "We only care for concept"
    if len(concepts) == 1:
        program = [
            "Filter",
            [
                "Scene"
            ],
            [
                concepts[0]
            ]
        ]
    else:
        program = [
            "Filter",
            [
                "Filter",
                [
                    "Scene"
                ],
                [
                    concepts[0]
                ]
            ],
            [
                concepts[1]
            ]
        ]
    return program

def get_stacked_scene_from_paths(image_path, annotation_path, dataset):
    image = load_image_from_path(image_path)
    mask, label, concept_name = get_mask_and_label_for_image_with_annotation_path(annotation_path, image)
    # index doesn't matter since we are not using cache
    transform = FixedResizeTransform(dataset.image_size)
    mask, image = transform(mask, image)
    stacked_scene = {
        "image": image, "mask": mask, "image_index": 0
    }
    return stacked_scene
def build_pretrain_instance(stacked_scene, dataset, concept_names):
    info = copy.deepcopy(dataset.info)
    concept_ids = [dataset.entry2idx_[c] for c in concept_names]
    program = build_program(build_question_program(concept_ids))
    instance = {
        **stacked_scene,
        'program': program,
        'info': info,
        'category': "choose_from_objects",
        'target': 0
    }
    return collate_fn([instance])

def build_fewshot_instance(train_stacked_scene, train_concept_name, relations, val_stacked_scene, dataset, val_concept_name):
    train_concept_index = dataset.entry2idx_[train_concept_name]
    train_stacked_scene = collate_fn([train_stacked_scene])
    train_program = [build_program(["Scene"])]
    train_samples = {
        **train_stacked_scene, 'program': train_program, 'image_index': [0], 'train_concept_index': train_concept_index
    }
    val_concept_index = [dataset.entry2idx_[v] for v in val_concept_name]
    val_stacked_scene = collate_fn([val_stacked_scene])
    val_program = build_question_program(val_concept_index)
    val_program = build_program(val_program)
    for vc in val_concept_index:
        val_program.register_token(vc)
    val_program = [val_program]
    val_samples = {
        **val_stacked_scene, 'program': val_program
    }

    relation_type = [r['relation_type'] for r in relations if r['relation_type'] != 'has_attribute']
    related_concept = [r['related_concept'] for r in relations if r['relation_type'] != 'has_attribute']
    supports = [dataset.entry2idx_[c] for c in related_concept]
    relations = [dataset.relation_entry2idx[r] for r in relation_type]
    metaconcept_program = build_program(("Composite", supports, relations, train_concept_index))
    info = copy.deepcopy(dataset.info)
    inputs = collate_fn([{
        'train_sample': train_samples, 'val_sample': val_samples, 'program': metaconcept_program, 'info': info, 'task': {}, 'concept_index': [train_concept_index], 'target': 0
    }])
    return inputs


def build_pretrain_instance_from_path(image_path, annotation_path, dataset, concept_names):
    stacked_scene = get_stacked_scene_from_paths(image_path, annotation_path, dataset)
    pretrain_inputs = build_pretrain_instance(stacked_scene, dataset, concept_names)
    return pretrain_inputs

def build_fewshot_instance_from_path(train_image_path, train_annotation_path, val_image_path, val_annotation_path, dataset, train_concept_name, val_concept_name):
    train_stacked_scene = get_stacked_scene_from_paths(train_image_path, train_annotation_path, dataset)
    val_stacked_scene = get_stacked_scene_from_paths(val_image_path, val_annotation_path, dataset)
    # suppose that we have the ground truth relations
    relations = dataset.relations[train_concept_name]
    fewshot_inputs = build_fewshot_instance(train_stacked_scene, train_concept_name, relations,val_stacked_scene, dataset, val_concept_name)
    return fewshot_inputs
def test(cfg, args):
    # Set seed for reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    logger = logging.getLogger("falcon_logger")
    logger.info("Setting up dependencies.")
    train_set = DatasetCatalog(cfg).get(cfg.DATASETS.TRAIN, cfg.DATASETS.DOMAIN, args)

    logger.info("Setting up models.")
    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    model = build_model(cfg).to("cuda")
    model = data_parallel(model, gpu_ids)
    # this line reset the box registry to the configuration that only pretrained concepts are in the concept net
    #model.reset_box_registry()
    model.eval()

    logger.info("Setting up utilities.")
    output_dir = cfg.OUTPUT_DIR

    # Weiwei's code for testing purpose
    # hardcoded to test
    logger.info("Setting up utilities.")
    output_dir = cfg.OUTPUT_DIR
    # load the weights
    checkpointer = Checkpointer(cfg, model, None, None)
    iteration = checkpointer.load(args.iteration)
    # known concept to keep track on the model's state
    known_concepts = {x:train_set.concepts_[x] for x in nonzero(train_set.concept2splits == -1)}

    pick_scene_root = "/home/local/ASUAD/weiweigu/data/test_dataset/House/pick_scenes"
    img_folder_root = join(pick_scene_root, 'images')
    annotation_folder_root = join(pick_scene_root, 'annotations')
    image_file = "images.txt"
    annotation_file = "annotations.txt"
    scene2obj_file = "scene2concept.txt"
    with open(join(pick_scene_root, image_file), 'r') as f:
        lines = f.readlines()
        image_list = [line.split(' ')[-1].rstrip('\n') for line in lines]
    with open(join(pick_scene_root, annotation_file), 'r') as f:
        lines = f.readlines()
        annotation_list = [line.split(' ')[-1].rstrip('\n') for line in lines]
    with open(join(pick_scene_root, scene2obj_file), 'r') as f:
        lines = f.readlines()
        obj_list = [line.split(' ')[-1].rstrip('\n') for line in lines]
    labels = []
    for anno in annotation_list:
        annotation_path = join(annotation_folder_root, anno)
        with open(annotation_path) as f:
            annotation = json.load(f)
        labels.append(annotation['label'])
    predictions = []

    for i, (img, anno, obj) in enumerate(zip(image_list, annotation_list, obj_list)):
        # when running inference on a known concept
        image_path = join(img_folder_root, img)
        annotation_file = join(annotation_folder_root, anno)

        pretrain_instance = build_pretrain_instance_from_path(image_path, annotation_file, train_set, concept_names=[obj])
        inputs = to_cuda(pretrain_instance)
        outputs = model.inference(inputs)
        logits = outputs['end'][0]
        predicted_label = logits.argmax(-1).squeeze()
        predictions.append(predicted_label)
        if obj == "yellow_square_tile":
            breakpoint()

    predictions = torch.stack(predictions)
    labels = torch.tensor(labels, device=predictions.device)
    total_correct_count = (predictions == labels).sum()
    split_info = torch.stack([train_set.concept2splits[train_set.entry2idx_[x]] for x in obj_list])
    pretrained_p = predictions[split_info == -1]
    trained_p = predictions[split_info == 0]
    val_p = predictions[split_info == 1]
    pretrained_l = labels[split_info == -1]
    trained_l = labels[split_info == 0]
    val_l = labels[split_info == 1]
    pretrained_correct_count = (pretrained_l == pretrained_p).sum()
    trained_correct_count = (trained_l == trained_p).sum()
    val_correct_count = (val_l == val_p).sum()
    breakpoint()




def main():
    parser = ArgumentParser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)
    output_dir = mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("falcon_logger", os.path.join(output_dir, "test_log.txt"))
    start_up()

    logger.info(f"Running with args:\n{args}")
    logger.info(f"Running with config:\n{cfg}")
    test(cfg, args)


if __name__ == "__main__":
    main()




