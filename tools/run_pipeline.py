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

import os
import torch
import logging
from transformers import BertTokenizer
from experiments.defaults import C
from dataset.utils import DataLoader, cycle
from dataset import NodeClassifierDataset
from models import NodeClassifierModel, ConceptExtractionModel
from solver import make_scheduler, Optimizer
from utils import to_cuda
from utils import setup_logger, start_up, ArgumentParser, FLAGS, data_parallel
from utils import Checkpointer, mkdir
from yacs.config import CfgNode as CN
import copy
from utils import SceneGraph, SceneGraphNode, convert_box_repr

from robot.generator import create_json_and_image
import cv2

#MODE = "knowledge"
MODE = "model"
# The path for synonym knowledge

SYNONYM_KNOWLEDGE_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/knowledge/tabletop_house_synonym.json"
# The path to the annotation for the user constructed structure
SCENE_ANNOTATION_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/robot/data/sample_scene.json"

# Path to the image and annotation that will be used for one-shot teaching concept
OBJECT_INSTANCE_IMAGE_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/robot/data/sample_one_shot.jpg"
# This file should also include the text made by the user when one-shotting a new concept
# E.g. This is a blue square block, it has the property of flooring and color of blue
OBJECT_INSTANCE_ANNOTATION_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/robot/data/sample_one_shot.json"

# Concept extractor
EXTRACTOR_CHECKPOINT_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/output/extraction_model/best_model.mdl"
EXTRACTOR_CONFIG = copy.deepcopy(C)

# Node classifier model path and configuration
# Please don't change the configuration
NODE_CLASSIFIER_CHECKPOINT_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/output/node_model/best_model.mdl"
NODE_CLASSIFIER_CONFIG = copy.deepcopy(C)


# Paths for pick/place scene in the loop
# These are the only paths for inputs that are changed during the inference loop
# Path to image and annotation file for pick scene
# (TODO: @Anant): Save the pick scene image and pick scene annotation to these paths
PICK_SCENE_IMAGE_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/robot/data/pick_scene.jpg"
PICK_SCENE_ANNOTATION_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/robot/data/pick_scene.json"

PLACE_SCENE_IMAGE_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/robot/data/sample_scene.jpg"
PLACE_SCENE_ANNOTATION_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/robot/data/sample_scene.json"
M = CN()
M.model_name = 'bert-base-cased'
M.MID_CHANNELS = 200
NODE_CLASSIFIER_CONFIG.M = M
EXTRACTOR_CONFIG.M = M

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

def load_image_from_path(image_path):
    t = TF.to_tensor(read_image(image_path))
    return t

def get_mask_for_image_with_annotation_path(annotation_path, image_tensor):
    # I am assuming that we use the same mask generation pipeline as the one that we used for getting the user demonstration scene
    with open(annotation_path) as f:
        annotation = json.load(f)
    bounding_boxes = [convert_box_repr(x) for x in annotation['bboxes']]
    image = image_tensor
    mask = torch.zeros(size=[len(bounding_boxes),image.shape[1], image.shape[2]])
    for h in range(mask.shape[1]):
        for w in range(mask.shape[2]):
            for i in range(len(bounding_boxes)):
                if (w >= bounding_boxes[i][0][0]) and (w <= bounding_boxes[i][0][1]) and (h >= bounding_boxes[i][1][0]) and (h <= bounding_boxes[i][1][1]):
                    mask[i, h, w] = 1
    mask = (mask > 0)
    return mask

def get_stacked_scene_from_paths(image_path, annotation_path, dataset):
    image = load_image_from_path(image_path)
    mask = get_mask_for_image_with_annotation_path(annotation_path, image)

    # index doesn't matter since we are not using cache
    transform = FixedResizeTransform(dataset.image_size)
    mask, image = transform(mask, image)
    # code for sanity check
    # x = image.permute((1, 2, 0)).numpy()
    # x = x * 255
    # x = x.astype(np.uint8)
    # x = Image.fromarray(x)
    # m = mask.permute((1, 2, 0)).numpy()

    #m = Image.fromarray(m)

    #m = mask.squeeze(0).numpy()
    #breakpoint()
    stacked_scene = {
        "image": image, "mask": mask, "image_index": 0
    }
    return stacked_scene

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


def build_fewshot_instance_from_path(train_image_path, train_annotation_path, val_image_path, val_annotation_path, dataset, train_concept_name, val_concept_name, relations):
    train_stacked_scene = get_stacked_scene_from_paths(train_image_path, train_annotation_path, dataset)
    val_stacked_scene = get_stacked_scene_from_paths(val_image_path, val_annotation_path, dataset)
    fewshot_inputs = build_fewshot_instance(train_stacked_scene, train_concept_name, relations,val_stacked_scene, dataset, val_concept_name)
    return fewshot_inputs

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


def build_pretrain_instance_from_path(image_path, annotation_path, dataset, concept_names):
    stacked_scene = get_stacked_scene_from_paths(image_path, annotation_path, dataset)
    pretrain_inputs = build_pretrain_instance(stacked_scene, dataset, concept_names)
    return pretrain_inputs



# (TODO: @Anant): change the find_concept_in_sentence method in your code to the followings

# ===========================================================================================
def find_concept_with_model(sentence, concept_candidates, model, tokenizer, candidate_type):
    negative_answer = f"No {candidate_type} is mentioned."
    candidates = [x.replace("_", " ") for x in concept_candidates]
    candidates = [*candidates, negative_answer]
    input_tokens = []
    input_masks = []
    for c in candidates:
        encoded_contextualized_sentence = tokenizer.encode_plus(
            sentence,
            c,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        )
        input_tokens.append(
                encoded_contextualized_sentence['input_ids']
        )
        input_masks.append(
                encoded_contextualized_sentence['attention_mask']
        )
    input_tokens = torch.stack(input_tokens, dim=1)
    input_masks = torch.stack(input_masks, dim=1)
    inputs = {
        "contextualized_sentences_tokens": input_tokens,
        "attention_mask": input_masks
    }
    inputs = to_cuda(inputs)
    outputs = model(inputs)
    if outputs['prediction'] == len(concept_candidates):
        # The model predicts none
        return None
    else:
        return concept_candidates[outputs['prediction']]
        

def find_concept_in_sentence_with_synonym_knowledge(sentence, concept_candidates):
    synonym_knowledge = load(SYNONYM_KNOWLEDGE_PATH)
    used_concepts = list(synonym_knowledge.keys())
    for c in concept_candidates:
        con = c.replace("_", " ")
        if con not in used_concepts:
            continue
        synonyms_of_concept = synonym_knowledge[con]
        found_match = False
        for sc in synonyms_of_concept:
            if sc in sentence:
                found_match = True
                return c
    return None



def find_concept_in_sentence(sentence, concept_candidates, model=None, tokenizer= None,candidate_type=None,mode="knowledge"):
    if mode == "knowledge":
        concept = find_concept_in_sentence_with_synonym_knowledge(sentence, concept_candidates)
    else:
        #(TODO: @Weiwei) implement an extraction with model
        concept = find_concept_with_model(sentence, concept_candidates, model, tokenizer, candidate_type)
    return concept


# ===========================================================================================


def run(cfg, args):
    # load the dataset for convenience
    train_set = DatasetCatalog(cfg).get(cfg.DATASETS.TRAIN, cfg.DATASETS.DOMAIN, args)

    # load synonym knowledge
    synonyms = load(SYNONYM_KNOWLEDGE_PATH)
    used_concepts = [x.replace(" ", "_") for x in synonyms.keys()]

    # load weights for concept net model
    gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    concept_net_model = build_model(cfg).to("cuda")
    concept_net_model = data_parallel(concept_net_model, gpu_ids)
    concept_net_model.eval()
    checkpointer = Checkpointer(cfg, concept_net_model, None, None)
    iteration = checkpointer.load(None)

    # ========================================================================
    # Resetting the box registry so that the model knows only about the pretrain concept
    # (TODO: @Anant) Check this line
    # ========================================================================

    concept_net_model.reset_box_registry()

    # Right now the known concepts are just pretrain concepts
    # We may adjust to add the train concepts

    known_concepts = [train_set.concepts_[x] for x in nonzero(train_set.concept2splits == -1)]
    test_concepts = [train_set.concepts_[x] for x in nonzero(train_set.concept2splits == 1)]

    # breakpoint()

    logger = logging.getLogger("falcon_logger")
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(
            NODE_CLASSIFIER_CONFIG.M.model_name
    )
    # breakpoint()
    # load weights for node classifier model
    node_classifier_model_state = torch.load(NODE_CLASSIFIER_CHECKPOINT_PATH)
    node_classifier_model = NodeClassifierModel(NODE_CLASSIFIER_CONFIG.M)
    node_classifier_model.load_state_dict(node_classifier_model_state["model"])
    node_classifier_model = node_classifier_model.to("cuda")
    node_classifier_model.eval()

    # Build the goal scene graph
    # completed with context

    # load weights for concept extractor model
    concept_extractor_model_state = torch.load(EXTRACTOR_CHECKPOINT_PATH)
    concept_extractor_model = ConceptExtractionModel(EXTRACTOR_CONFIG.M)
    concept_extractor_model.load_state_dict(concept_extractor_model_state["model"])
    concept_extractor_model = concept_extractor_model.to("cuda")
    concept_extractor_model.eval()
    # take user query, this should be pre-recorded in some path, or place it along with the scene
    # hard coded for now
    user_query = "Build me a house where the floor is yellow."

    with open(SCENE_ANNOTATION_PATH, "r") as f:
        annotation = json.load(f)
    user_input_scene_graph = SceneGraph.from_annotated_scene(annotation)
    object_concepts = train_set.objects_
    all_concepts = train_set.concepts_[:-2]
    non_leaf_concepts = list(set(all_concepts).difference(object_concepts))
    affordances = ['supporting', 'flooring', 'sheltering']
    colors = list(set(non_leaf_concepts).difference(affordances))

    # =======================================================================================
    # Build the goal scene graph
    # =======================================================================================
    goal_scene_graph = SceneGraph("house")
    object_concepts = list(set(object_concepts).intersection(used_concepts))

    leaf_concept_from_user_query = find_concept_in_sentence(user_query, object_concepts, model=concept_extractor_model, tokenizer=tokenizer, candidate_type = "object", mode=MODE)
    concept_from_user_query = None

    if leaf_concept_from_user_query == None:
        concept_from_user_query = find_concept_in_sentence(user_query, colors, model=concept_extractor_model, tokenizer=tokenizer, candidate_type = "color", mode=MODE)
    else:
        concept_from_user_query = leaf_concept_from_user_query

    # The user query should give indication on some concepts
    assert concept_from_user_query != None, "The user query does not provide sufficient information!"

    # breakpoint()
    for node in user_input_scene_graph.nodes:

        if node.node_id == 0:
            # ground node
            new_ground_node = copy.deepcopy(node)
            goal_scene_graph.nodes.append(new_ground_node)

        else:
            new_node_dict = {
                "node_id": copy.deepcopy(node.node_id),
                "positional_relations": copy.deepcopy(node.positional_relations),
                "higher_level_concept": None,
                "node_context": copy.deepcopy(node.node_context),
                "bounding_box": None
            }

            node_context = node.node_context
            block_from_node_context = find_concept_in_sentence(node_context, object_concepts, model=concept_extractor_model, tokenizer=tokenizer, candidate_type = "object", mode=MODE)
            affordance_from_node_context = find_concept_in_sentence(node_context, ['supporting', 'flooring', 'sheltering'], model=concept_extractor_model, tokenizer=tokenizer, candidate_type = "affordance", mode=MODE)
            # The node context from the user should always convey these two information, this is just a sanity check
            # Disabled for now because it doesn't in the current test file
            #assert block_from_node_context != None, "User's description does not give information about which object is used for this node!"
            #assert affordance_from_node_context != None, "User's description does not provide information about the affordance of the object for the node!"

            encoded_contextualized_query = tokenizer.encode_plus(
                node_context,
                user_query,
                add_special_tokens=True,
                max_length=512,
                return_attention_mask=True,
                pad_to_max_length=True,
                truncation=True,
                return_tensors='pt'
            )

            inputs_to_node_classifier = {
                "contextualized_query_tokens": encoded_contextualized_query['input_ids'],
                "contextualized_query_attention_mask": encoded_contextualized_query['attention_mask']
            }
            inputs_to_node_classifier = to_cuda(inputs_to_node_classifier)
            node_classifier_prediction = node_classifier_model(inputs_to_node_classifier)['prediction']
            if node_classifier_prediction:
                if leaf_concept_from_user_query == None:
                    # The user's query is directly about object
                    block_type = [concept_from_user_query, affordance_from_node_context]
                else:
                    block_type = [concept_from_user_query]
            else:
                # use the extracted block_type from the node context
                block_type = [block_from_node_context]
            new_node_dict["block_type"] = block_type
            new_node = SceneGraphNode.from_dict(new_node_dict)
            goal_scene_graph.nodes.append(new_node)
    # ================================================================================================================
    # FINISH CONSTRUCTING GOAL SCENE GRAPH ABOVE
    # ================================================================================================================


    # ================================================================================================================
    # Update the network by one-shotting
    # ================================================================================================================

    # This sentence should be loaded from the annotation of the one-shot file
    one_shot_annotation = load(OBJECT_INSTANCE_ANNOTATION_PATH)

    one_shot_sentence = one_shot_annotation['text']
    one_shot_object = find_concept_in_sentence(one_shot_sentence, object_concepts, model=concept_extractor_model, tokenizer=tokenizer, candidate_type = "object", mode=MODE)
    one_shot_color = find_concept_in_sentence(one_shot_sentence, colors, model=concept_extractor_model, tokenizer=tokenizer, candidate_type = "color", mode=MODE)
    one_shot_affordance = find_concept_in_sentence(one_shot_sentence, affordances, model=concept_extractor_model, tokenizer=tokenizer, candidate_type = "affordance", mode=MODE)




    assert one_shot_object != None, "The user sentence does not mention an object!"
    relations = []
    if one_shot_color is not None:
        relations.append(
            {
                "relation_type": "has_color",
                "related_concept": one_shot_color
            }
        )
    if one_shot_affordance is not None:
        relations.append(
            {
                "relation_type": "has_affordance",
                "related_concept": one_shot_affordance
            }
        )
    # build one-shot instance just to insert the concept
    # the val image/annotation doesn't matter as we just want to update the embedding
    one_shot_instance = build_fewshot_instance_from_path(train_image_path=OBJECT_INSTANCE_IMAGE_PATH, train_annotation_path=OBJECT_INSTANCE_ANNOTATION_PATH, val_image_path=OBJECT_INSTANCE_IMAGE_PATH, val_annotation_path=OBJECT_INSTANCE_ANNOTATION_PATH,
                                    dataset=train_set, train_concept_name=one_shot_object, val_concept_name=[one_shot_object], relations=relations)
    
    oneshot_inputs = to_cuda(one_shot_instance)
    oneshot_outputs = concept_net_model.inference(oneshot_inputs)
    # don't forget to use concept_net_model.callback() to update the embeddings

    
    concept_net_model.update_concept_embedding(oneshot_outputs['queried_embedding'])

    # breakpoint()
    # Do pick and place for each node
    
    
    COUNT=0
    # breakpoint()
    for node in goal_scene_graph.nodes:
        if node.node_id == 0:
            # This is just ground, skip
            continue
        # (TODO: @Anant): Take image and generate mask for pick scene here
        
        # x=input("This is a checker ")
        # create_json_and_image(PICK_SCENE_IMAGE_PATH,PICK_SCENE_ANNOTATION_PATH,0,2)
        # print(COUNT)
        
        # image()
        if node.block_type is None or len(node.block_type)==0:
            breakpoint()
        pick_scene_inputs = build_pretrain_instance_from_path(PICK_SCENE_IMAGE_PATH, PICK_SCENE_ANNOTATION_PATH, dataset=train_set, concept_names=node.block_type)
        #pick_scene_inputs = build_pretrain_instance_from_path(OBJECT_INSTANCE_IMAGE_PATH, OBJECT_INSTANCE_ANNOTATION_PATH, dataset=train_set, concept_names=node.block_type)
        pick_scene_inputs = to_cuda(pick_scene_inputs)
        pick_scene_outputs = concept_net_model.inference(pick_scene_inputs)
        # print(pick_scene_outputs)
        # breakpoint()

        pick_scene_logits = pick_scene_outputs['end'][0]
        # print(pick_scene_logits)
        predicted_object_index = pick_scene_logits.argmax(-1).squeeze()

        # retrieve the bounding box from the predicted index
        pick_scene_annotation = load(PICK_SCENE_ANNOTATION_PATH)

        predicted_bb = pick_scene_annotation['bboxes'][predicted_object_index]
        breakpoint()
        # print(predicted_bb)
        
        STR="./debug/debug_"+str(COUNT)+".jpg"
        

        image=cv2.imread(PICK_SCENE_IMAGE_PATH)
        
        cv2.rectangle(image,(predicted_bb[0],predicted_bb[1]),((predicted_bb[0]+predicted_bb[2]),(predicted_bb[1]+predicted_bb[3])),(255,0,0), 2)
        cv2.imwrite(STR,image)
        
        COUNT=COUNT+1

        # x=input("Break: ")







        #(TODO: @Anant): grasp from the predicted bounding box

        #(TODO: @Anant): Compute the place location based on the relationship of the current node, and the bounding box of the previous node

        #(TODO: @Anant): set the location for this node after placing








def main():
    parser = ArgumentParser()
    args = parser.parse_args()
    # breakpoint()
    # print("*********************************")
    cfg = cfg_from_args(args)
    output_dir = mkdir(cfg.OUTPUT_DIR)
    # print("here**********************")
    logger = setup_logger("falcon_logger", os.path.join(output_dir, "test_log.txt"))
    # print("here")
    start_up()
    #
    # logger.info(f"Running with args:\n{args}")
    # logger.info(f"Running with config:\n{cfg}")
    run(cfg, args)

if __name__ == "__main__":
    main()
