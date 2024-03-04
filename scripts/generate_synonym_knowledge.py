from utils import load
import json

OBJECT_PATH = "/home/local/ASUAD/weiweigu/data/test_dataset/.augmented/House/0/tabletop/objects.json"
SYNONYM_KNOWLEDGE_PATH = "/home/local/ASUAD/weiweigu/research/FALCON-Generalized/knowledge/tabletop_house_synonym.json"


def main():
    colors = {
        "blue": [
            "blue",
            "azure",
            "cobalt",
            "sapphire",
            "indigo"
        ], 
        "green": [
            "green",
            "lime"
        ], 
        "orange": [
            "orange",
            "gold",
            "golden"
            
        ], 
        "red": [
            "red",
            "scarlet",
            "ruby",
            "vermilion",
            "cherry"
        ], 
        "wood": [
            "wood",
            "straw",
            "brown",
            "lumber",
            "wooden"
        ], 
        "yellow": [
            "yellow",
            "creamy"
        ]
    }

    shapes = {
        "curve": [
            "curve",
            "curvy",
            "curves",
            "curved",
            "sheltering",
            "cover",
            "shelter",
            "roofing",
            "shield"
        ],
        "short rectangular": [
            "short rectangular",
            "rectangular",
            "rectangle",
            "short rectangle",
            "supporting",
            "support",
            "strength",
            "reinforcing"
        ],
        "square": [
            "square",
            "squared",
            "squares",
            "flooring",
            "paving",
            "tiling",
            "floor",
            "pave",
            "ground"
        ]
    }
    
    affordances = {
        "supporting": [
            "supporting",
            "support",
            "strength",
            "reinforcing"
        ],
        "flooring": [
            "flooring",
            "paving",
            "tiling",
            "floor",
            "pave",
            "ground"
        ],
        "sheltering": [
            "sheltering",
            "cover",
            "shelter",
            "roofing",
            "shield"
        ]
    }
    objects = load(OBJECT_PATH)
    shape_names = {
        "curve block": "curve",
        "short rectangular tile": "short rectangular",
        "square tile": "square"
    }
    obj_dict = {}
    for obj in objects:
        obj_name = obj.replace("_", " ")
        obj_used = False
        for sn in shape_names.keys():
            if sn in obj_name:
                obj_used = True
                sn_name = shape_names[sn]
                break
        if not(obj_used):
            continue
        synonyms_for_current_object = []

        object_color_synonyms = colors[obj.split("_")[0]]
        object_shape_synonyms = shapes[sn_name]

        for ocs in object_color_synonyms:
            for oss in object_shape_synonyms:
                synonyms_for_current_object.extend([f"{ocs} {oss}", f"{oss} {ocs}"])

        obj_dict[obj_name] = synonyms_for_current_object

    synonyms = {**obj_dict, **colors, **affordances}
    with open(SYNONYM_KNOWLEDGE_PATH, 'w') as f:
        json.dump(synonyms, f, indent=4)
    


                

if __name__ == "__main__":
    main()

