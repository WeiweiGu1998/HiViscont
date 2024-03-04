import json
from utils import load, join, nonzero
import random
import copy
import torch


DATASET_ROOT = "/home/local/ASUAD/weiweigu/data/test_dataset"
DOMAIN = "House"
SPLIT_SEED = 103
PROB_OF_SYNONYM = 0.3
DATASET_AUGMENTED_ROOT = "/home/local/ASUAD/weiweigu/data/test_dataset/.augmented"
OBJECT_FILE = join(f"{DATASET_AUGMENTED_ROOT}/{DOMAIN}/{SPLIT_SEED}/tabletop", "objects.json")
CONCEPT_FILE = join(f"{DATASET_AUGMENTED_ROOT}/{DOMAIN}/{SPLIT_SEED}/tabletop", "concepts.json")
CONCEPT_SPLIT_FILE = join(f"{DATASET_AUGMENTED_ROOT}/{DOMAIN}/{SPLIT_SEED}/tabletop", "concept_split.json")
RELATION_FILE = f'knowledge/tabletop_{DOMAIN.lower()}_relations.json'

DATA_PATH = "/home/local/ASUAD/weiweigu/Desktop/small_dataset.json"
SYNONYMS = {
    "roof": ["ceiling", "covering", "top"],
    "pillar": ["column", "post", "support", "pile"],
    "floor": ["ground", "deck"]
}
ORIGINAL_SCENE_KNOWLEDGE = {
    "house": {
        "roof": "sheltering",
        "pillar": "supporting",
        "floor": "flooring",
        # "roof": [0, 6, 11, 26],
        # "pillar": [1, 2, 3, 7, 8, 12, 13, 16, 17, 18, 21, 22, 23, 27, 28],
        # "floor": [4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 30],
    }
}
NODE_CONTEXT_TEMPLATES = [
    "I build a [SCENE_ROLE] with [OBJECT_0], because it has the property of [AFFORDANCE].",
    "The [SCENE_ROLE] is build by the [OBJECT_0] because of its [AFFORDANCE] property",
    "The [OBJECT_0] has the [AFFORDANCE] property. So, we build the [SCENE_ROLE] with it."
]
TEMPLATES = {
    # "removal": [
    #     "Build the [SCENE] without [SCENE_ROLE].",
    #     "Build the [SCENE] with no [SCENE_ROLE].",
    #     "Remove the [SCENE_ROLE] from the [SCENE]."
    # ],
    "replacement": [
        "Build me a [COLOR] [SCENE_ROLE].",
        "Build a [COLOR] [SCENE_ROLE].",
        "Build a [SCENE] with a [COLOR] [SCENE_ROLE].",
        "Build the [SCENE] with [OBJECT_1].",
        "Build the [SCENE_ROLE] with [OBJECT_1].",
        "Replace [OBJECT_0] with [OBJECT_1].",
        "Use [OBJECT_1] to build the [SCENE_ROLE]."
    ]
}
def generate_positive_samples(scene, curr_split, relations, concept_split_spec, entry2idx):
    SCENE_KNOWLEDGE = copy.deepcopy(ORIGINAL_SCENE_KNOWLEDGE)
    idx2entry = {v:k for k,v in entry2idx.items()}
    #entry2idx["empty"] = len(concept_split_spec)
    # sample scene role
    scene_role = random.sample(list(SCENE_KNOWLEDGE[scene].keys()),1)[0]
    chosen_affordance = SCENE_KNOWLEDGE[scene][scene_role]
    # get concept candidate for the scene role
    candidates_of_scene = []
    for o, r in relations.items():
        related_concepts = [re["related_concept"] for re in r]
        if chosen_affordance in related_concepts:
            candidates_of_scene.append(o)

    valid_prev_concepts = [idx2entry[x] for x in nonzero(concept_split_spec < 1)]
    valid_prev_concepts = [x for x in valid_prev_concepts if (('short_rectangular_tile' in x) or ('square_tile' in x) or ('curve_block' in x))]
    candidates_prev_concepts = list(set(valid_prev_concepts).intersection(candidates_of_scene))
    if curr_split == "train":
        valid_curr_concepts = [idx2entry[x] for x in nonzero(concept_split_spec < 1)]
    else:
        valid_curr_concepts = [idx2entry[x] for x in nonzero(concept_split_spec == 1)]

    valid_curr_concepts = [x for x in valid_curr_concepts if
                           (('short_rectangular_tile' in x) or ('square_tile' in x) or ('curve_block' in x))]
    original_concept = random.sample(candidates_prev_concepts, 1)[0]
    original_concept_color = [r['related_concept'] for r in relations[original_concept] if r['relation_type'] == "has_color"][0]
    candidates_curr_concepts = list(set(valid_curr_concepts).intersection(candidates_of_scene) - set([original_concept]))
    new_concept = random.sample(candidates_curr_concepts, 1)[0]
    new_concept_color = \
    [r['related_concept'] for r in relations[new_concept] if r['relation_type'] == "has_color"][0]
    scene_role_0 = scene_role
    scene_role_1 = scene_role
    x_0 = random.random()
    x_1 = random.random()
    if x_0 < PROB_OF_SYNONYM:
        scene_role_0 = random.sample(SYNONYMS[scene_role], 1)[0]
    if x_1 < PROB_OF_SYNONYM:
        scene_role_1 = random.sample(SYNONYMS[scene_role], 1)[0]
    node_context = random.sample(NODE_CONTEXT_TEMPLATES, 1)[0]
    node_context = node_context.replace('[SCENE_ROLE]', scene_role_0)
    textual_original_concept = original_concept.replace("_", " ")
    node_context = node_context.replace('[OBJECT_0]', textual_original_concept)
    node_context = node_context.replace('[AFFORDANCE]', chosen_affordance)
    query = random.sample(TEMPLATES["replacement"], 1)[0]
    textual_new_concept = new_concept.replace("_", " ")
    query = query.replace('[SCENE_ROLE]', scene_role_1)
    query = query.replace('[OBJECT_0]', textual_original_concept)
    query = query.replace('[OBJECT_1]', textual_new_concept)
    query = query.replace('[COLOR]', new_concept_color)
    query = query.replace('[SCENE]', "house")
    sample = {
        "query": query,
        "node_context": node_context,
        "scene_role_of_node": scene_role_0,
        "scene_role_of_query": scene_role_1,
        "affordance_of_node": chosen_affordance,
        "affordance_of_query": chosen_affordance,
        "node_concept": original_concept,
        "node_concept_color": original_concept_color,
        "original_concept": original_concept,
        "original_concept_color": original_concept_color,
        "query_concept": new_concept,
        "query_concept_color": new_concept_color,
        "label": True
    }
    return sample

def generate_negative_samples(scene, curr_split, relations, concept_split_spec, entry2idx):
    SCENE_KNOWLEDGE = copy.deepcopy(ORIGINAL_SCENE_KNOWLEDGE)
    idx2entry = {v: k for k, v in entry2idx.items()}
    # entry2idx["empty"] = len(concept_split_spec)
    # sample scene role
    scene_roles = random.sample(list(SCENE_KNOWLEDGE[scene].keys()), 2)
    node_scene_role = scene_roles[0]
    query_scene_role = scene_roles[1]

    node_affordance = SCENE_KNOWLEDGE[scene][node_scene_role]
    query_affordance = SCENE_KNOWLEDGE[scene][query_scene_role]


    # get concept candidate for the scene role
    candidates_of_query = []
    candidates_of_node = []
    for o, r in relations.items():
        related_concepts = [re["related_concept"] for re in r]
        if node_affordance in related_concepts:
            candidates_of_node.append(o)
        if query_affordance in related_concepts:
            candidates_of_query.append(o)

    valid_node_concepts = [idx2entry[x] for x in nonzero(concept_split_spec < 1)]
    valid_node_concepts = [x for x in valid_node_concepts if
                           (('short_rectangular_tile' in x) or ('square_tile' in x) or ('curve_block' in x))]
    candidates_node_concepts = list(set(valid_node_concepts).intersection(candidates_of_node))
    if curr_split == "train":
        valid_query_concepts = [idx2entry[x] for x in nonzero(concept_split_spec < 1)]
    else:
        valid_query_concepts = [idx2entry[x] for x in nonzero(concept_split_spec == 1)]

    valid_query_concepts = [x for x in valid_query_concepts if
                           (('short_rectangular_tile' in x) or ('square_tile' in x) or ('curve_block' in x))]
    node_concept = random.sample(candidates_node_concepts, 1)[0]
    node_concept_color = \
    [r['related_concept'] for r in relations[node_concept] if r['relation_type'] == "has_color"][0]
    candidates_query_concepts = list(
        set(valid_query_concepts).intersection(candidates_of_query))
    query_concept = random.sample(candidates_query_concepts, 1)[0]
    query_concept_color = \
        [r['related_concept'] for r in relations[query_concept] if r['relation_type'] == "has_color"][0]
    node_context = random.sample(NODE_CONTEXT_TEMPLATES, 1)[0]
    node_x = random.random()
    query_x = random.random()
    if node_x < PROB_OF_SYNONYM:
        node_scene_role = random.sample(SYNONYMS[node_scene_role], 1)[0]
    if query_x < PROB_OF_SYNONYM:
        query_scene_role = random.sample(SYNONYMS[query_scene_role], 1)[0]
    node_context = node_context.replace('[SCENE_ROLE]', node_scene_role)
    textual_node_concept = node_concept.replace("_", " ")
    node_context = node_context.replace('[OBJECT_0]', textual_node_concept)
    node_context = node_context.replace('[AFFORDANCE]', node_affordance)
    query = random.sample(TEMPLATES["replacement"], 1)[0]
    textual_query_concept = query_concept.replace("_", " ")
    query = query.replace('[SCENE_ROLE]', query_scene_role)
    candidates_original_concepts = list(
        set(valid_node_concepts).intersection(candidates_of_query) - set([query_concept]))
    original_concept = random.sample(candidates_original_concepts, 1)[0]
    original_concept_color = \
    [r['related_concept'] for r in relations[original_concept] if r['relation_type'] == "has_color"][0]
    textual_original_concept = original_concept.replace("_", " ")
    query = query.replace('[OBJECT_0]', textual_original_concept)
    query = query.replace('[OBJECT_1]', textual_query_concept)
    query = query.replace('[COLOR]', query_concept_color)
    query = query.replace('[SCENE]', "house")
    sample = {
        "query": query,
        "node_context": node_context,
        "scene_role_of_node": node_scene_role,
        "scene_role_of_query": query_scene_role,
        "affordance_of_node": node_affordance,
        "affordance_of_query": query_affordance,
        "node_concept": node_concept,
        "node_concept_color": node_concept_color,
        "original_concept": original_concept,
        "original_concept_color": original_concept_color,
        "query_concept": query_concept,
        "query_concept_color": query_concept_color,
        "label": False
    }
    return sample



def main():
    number_to_generate = 1000
    concepts_ = load(CONCEPT_FILE)
    concepts = list(range(len(concepts_)))
    concept_split_specs = torch.tensor(load(CONCEPT_SPLIT_FILE))
    concept2splits = (concept_split_specs - 1).float().div(3).clamp(min=-1.,max=1.).floor()
    objects_ = load(OBJECT_FILE)
    objects = list(range(len(objects_)))
    data = []
    # concepts, leaf_concepts, non_leaf_concepts = build_concepts(CONCEPT_KNOWLEDGE_PATH)
    # concepts_ = list(range(len(concepts)))
    entry2idx = {
        c:i for i,c in enumerate(concepts_)
    }
    relations = load(RELATION_FILE)
    affordances = [31, 32, 33]
    colors = [34, 35, 36, 37, 38, 39]

    train_concepts = nonzero(concept2splits < 1)
    val_concepts = nonzero(concept2splits == 1)

    chance_for_pos = 0.5
    for i in range(7000):
        if random.random() < chance_for_pos:
            sample = generate_positive_samples(scene="house", curr_split="train", relations=relations, concept_split_spec=concept2splits, entry2idx=entry2idx)
        else:
            #sample = generate_negative_samples(scene="house", curr_split=0, hyper2hypo=hyper2hypo, concept_split_spec=split_spec, entry2idx=entry2idx)
            sample = generate_negative_samples(scene="house", curr_split="train", relations=relations,
                                               concept_split_spec=concept2splits, entry2idx=entry2idx)
        data.append(sample)

    for i in range(3000):
        if random.random() < chance_for_pos:
            sample = generate_positive_samples(scene="house", curr_split="val", relations=relations, concept_split_spec=concept2splits, entry2idx=entry2idx)
        else:
            #sample = generate_negative_samples(scene="house", curr_split=0, hyper2hypo=hyper2hypo, concept_split_spec=split_spec, entry2idx=entry2idx)
            sample = generate_negative_samples(scene="house", curr_split="val", relations=relations,
                                               concept_split_spec=concept2splits, entry2idx=entry2idx)
        data.append(sample)


    # for i in range(100):
    #     if random.random() < chance_for_pos:
    #         sample = generate_positive_samples(scene="house", curr_split=2, hyper2hypo=hyper2hypo,
    #                                            concept_split_spec=split_spec, entry2idx=entry2idx)
    #     else:
    #         sample = generate_negative_samples(scene="house", curr_split=2, hyper2hypo=hyper2hypo,
    #                                            concept_split_spec=split_spec, entry2idx=entry2idx)
    #     data.append(sample)
    #
    with open(DATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()