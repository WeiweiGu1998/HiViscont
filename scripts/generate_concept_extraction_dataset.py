import json
from utils import load, join, nonzero
import random
import copy
import torch


SYNONYM_KNOWLEDGE_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/knowledge/tabletop_house_synonym.json"
PROB_FOR_COLOR = 0.3
PROB_FOR_AFFORDANCE = 0.3
PROB_FOR_OBJECT = 0.4
DATASET_ROOT = "/home/local/ASUAD/weiweigu/data/test_dataset"
DOMAIN = "House"
SPLIT_SEED = 103
DATASET_AUGMENTED_ROOT = "/home/local/ASUAD/weiweigu/data/test_dataset/.augmented"
OBJECT_FILE = join(f"{DATASET_AUGMENTED_ROOT}/{DOMAIN}/{SPLIT_SEED}/tabletop", "objects.json")
CONCEPT_FILE = join(f"{DATASET_AUGMENTED_ROOT}/{DOMAIN}/{SPLIT_SEED}/tabletop", "concepts.json")
CONCEPT_SPLIT_FILE = join(f"{DATASET_AUGMENTED_ROOT}/{DOMAIN}/{SPLIT_SEED}/tabletop", "concept_split.json")
RELATION_FILE = f'knowledge/tabletop_{DOMAIN.lower()}_relations.json'

DATA_PATH = "/home/local/ASUAD/weiweigu/Desktop/extraction_dataset.json"
SYNONYMS = {
    "roof": ["ceiling", "covering", "top", "cover", "shelter"],
    "pillar": ["column", "post", "support"],
    "floor": ["ground", "deck", "base"]
}

COLORS = ["blue", "green", "orange", "red", "wood", "yellow"]
AFFORDANCES = ["supporting", "sheltering", "flooring"]

BLOCK_SYNONYM = ["tile", "block"]

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


ONESHOT_SENTENCE_TEMPLATES = [
    "This is a [OBJECT_0]. It has the property of [AFFORDANCE] and it is [COLOR].",
    "This is a [OBJECT_0]. It is [COLOR] and can be used for [AFFORDANCE].",
    "This is a [OBJECT_0]. It has the property of [AFFORDANCE] and it is [COLOR].",
    "The [OBJECT_0] is [COLOR] and has the property of [AFFORDANCE]."
]

NODE_CONTEXT_TEMPLATES = [
    "I build a [SCENE_ROLE] with [OBJECT_0], because it has the property of [AFFORDANCE].",
    "The [SCENE_ROLE] is build by the [OBJECT_0] because of its [AFFORDANCE] property",
    "The [OBJECT_0] has the [AFFORDANCE] property. So, we build the [SCENE_ROLE] with it."
]
QUERY_TEMPLATES = {
    "replacement": [
        "Build me a [COLOR] [SCENE_ROLE].",
        "Build a [COLOR] [SCENE_ROLE].",
        "Build a [SCENE] with a [COLOR] [SCENE_ROLE].",
        "Build the [SCENE_ROLE] with [OBJECT_0].",
    ]
}

def generate_samples(object_candidates, relations, synonyms,candidate_type):
    negative_answer = f"No {candidate_type} is mentioned."
    true_object = random.choices(object_candidates, k=1)[0]
    true_object_rels = relations[true_object]
    true_object_color = [x["related_concept"] for x in true_object_rels if x["relation_type"] == "has_color"][0]
    true_object_affordance = [x["related_concept"] for x in true_object_rels if x["relation_type"] == "has_affordance"][0]
    affordance2role = {
        "flooring": "floor",
        "supporting": "pillar",
        "sheltering": "roof"
    }
    role = affordance2role[true_object_affordance]
    sentence_type = None 

    if candidate_type == "affordance":
        candidates = AFFORDANCES
        sentence_type = random.choices([0,1],k=1)[0]
    elif candidate_type == "object":
        candidates = [x.replace("_", " ") for x in object_candidates]
        sentence_type = random.choices([0,1,2],k=1)[0]
    elif candidate_type == "color":
        candidates = COLORS
        sentence_type = random.choices([0,2],k=1)[0]
        
    templates = None
    if sentence_type == 0:
        templates = ONESHOT_SENTENCE_TEMPLATES
    elif sentence_type == 1:
        templates = NODE_CONTEXT_TEMPLATES
    else:
        templates = QUERY_TEMPLATES["replacement"]

    sentence = random.choices(templates,k=1)[0]
    sentence = sentence.replace("[SCENE]", "house")
    sentence = sentence.replace("[SCENE_ROLE]", role)

    answer_candidates = [*candidates, negative_answer]
    true_answer = len(answer_candidates) - 1
    if candidate_type == "object":
        if "[OBJECT_0]" in sentence:
            true_answer = answer_candidates.index(true_object.replace("_", " "))
    elif candidate_type == "color":
        if "[COLOR]" in sentence:
            true_answer = answer_candidates.index(true_object_color)
    elif candidate_type == "affordance":
        if "[AFFORDANCE]" in sentence:
            true_answer = answer_candidates.index(true_object_affordance)
    
    object_word = random.choices(synonyms[true_object.replace("_", " ")], k=1)[0]
    block_name = ""
    if "curve" in true_object:
        block_name = random.choices(BLOCK_SYNONYM, weights=[2, 8], k=1)[0]
    else:
        block_name = random.choices(BLOCK_SYNONYM, weights=[8, 2], k=1)[0]
    object_word = f"{object_word} {block_name}"
    color_word = random.choices(synonyms[true_object_color], k=1)[0]
    affordance_word = random.choices(synonyms[true_object_affordance], k=1)[0]
    sentence = sentence.replace("[OBJECT_0]", object_word)
    sentence = sentence.replace("[COLOR]", color_word)
    sentence = sentence.replace("[AFFORDANCE]", affordance_word)

    sample = {
        "candidates": answer_candidates,
        "sentence": sentence,
        "label":true_answer 
    }
    return sample


def main():
    number_to_generate = 1000
    synonyms = load(SYNONYM_KNOWLEDGE_PATH) 
    concepts_ = load(CONCEPT_FILE)
    concepts = list(range(len(concepts_)))
    concept_split_specs = torch.tensor(load(CONCEPT_SPLIT_FILE))
    concept2splits = (concept_split_specs - 1).float().div(3).clamp(min=-1.,max=1.).floor()
    objects_ = load(OBJECT_FILE)
    objects = list(range(len(objects_)))
    data = []

    used_objects = [y.replace(" ", "_") for y in list(set([x.replace(" ", "_") for x in synonyms.keys()]).intersection(objects_))]

    # concepts, leaf_concepts, non_leaf_concepts = build_concepts(CONCEPT_KNOWLEDGE_PATH)
    # concepts_ = list(range(len(concepts)))
    entry2idx = {
        c:i for i,c in enumerate(concepts_)
    }
    relations = load(RELATION_FILE)

    train_concepts = nonzero(concept2splits < 1)
    train_concepts_ = [concepts_[x] for x in train_concepts] 
    val_concepts = nonzero(concept2splits == 1)
    val_concepts_ = [concepts_[x] for x in val_concepts] 

    for i in range(7000):
        valid_object_concepts = list(set(used_objects).intersection(train_concepts_))
        mode = random.choices(['object', 'color', 'affordance'], weights = [0.4, 0.3, 0.3], k=1)[0]
        sample = generate_samples(object_candidates=valid_object_concepts, relations=relations, synonyms=synonyms,candidate_type=mode)
        data.append(sample)

    for i in range(3000):
        valid_object_concepts = used_objects
        mode = random.choices(['object', 'color', 'affordance'], weights = [0.4, 0.3, 0.3], k=1)[0]
        sample = generate_samples(object_candidates=valid_object_concepts, relations=relations, synonyms=synonyms,candidate_type=mode)
        data.append(sample)


    with open(DATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
