import torch
from torch import nn
import copy
import json

HORIZONTAL_DISTANCE_RANGE = 36
VERTICAL_DISTANCE_RANGE = 0.75 * HORIZONTAL_DISTANCE_RANGE

def get_box_from_points(points):
    x_s = [p[0] for p in points]
    y_s = [p[1] for p in points]
    return [(min(x_s), max(x_s)),(min(y_s), max(y_s))]

def convert_box_repr(box):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    return [(x, x+w),(y, y+h)]

def get_loc_from_boxes(boxes):
    bb = convert_box_repr(boxes)
    return ((bb[0][0] + bb[0][1])/2,(bb[1][0] + bb[1][1])/2)

def get_loc_from_points(points):
    bb = get_box_from_points(points)
    return ((bb[0][0] + bb[0][1])/2,(bb[1][0] + bb[1][1])/2)

def check_overlap(r_1, r_2):
    # give a 10 pixel cushion
    if r_1[0] <= r_2[0]:
        if r_1[1] >= r_2[0] + 10:
            return True
        else:
            return False
    else:
        if r_2[1] >= r_1[0] + 10:
            return True
        else:
            return False

class SceneGraphNode:
    """
    This class is a node of the scene graph.
    Params:
        block_type (int):
            The index of the block in the concept net
        higher_level_concept (int):
            Describes the function of the node, e.g. floor/pillar/roof
        positional_relations list((int, int)):
            Describes the relations with other blocks
    """
    def __init__(
        self,
        node_id,
        block_type,
        higher_level_concept,
        positional_relations,
        node_context,
        bounding_box
    ):
        self.node_id = node_id
        self.block_type = block_type
        self.higher_level_concept = higher_level_concept
        self.positional_relations = positional_relations
        self.node_context = node_context
        self.bounding_box = bounding_box

    @staticmethod
    def from_dict(d):
        scene_graph_node = SceneGraphNode(
            node_id=d["node_id"],
            block_type=d["block_type"],
            higher_level_concept=d["higher_level_concept"],
            positional_relations=d["positional_relations"],
            bounding_box=d["bounding_box"],
            node_context=d["node_context"]
        )
        return scene_graph_node

    def __str__(self):
        return str(self.__dict__)


class SceneGraph:
    

    def __init__(self, name):
        self.name = name
        self.nodes = []

    @staticmethod
    def from_annotated_scene(
        annotation
    ):
        """
        Construct a SceneGraph from a scene.
        Params:
            annotation (dict):
                The annotation of a scene, including the image, a list of objects along with bounding boxes
        """
        #object_list = annotation["shapes"]
        # sort such that the order is from left-to-right and from bottom-to-top.
        #object_list = sorted(object_list, key=lambda x: (-get_loc_from_points(x["points"])[1]//VERTICAL_DISTANCE_RANGE, get_loc_from_points(x["points"])[0]//HORIZONTAL_DISTANCE_RANGE))
        object_list = annotation["bboxes"]
        object_list = sorted(object_list, key=lambda x: (
        -get_loc_from_boxes(x)[1] // VERTICAL_DISTANCE_RANGE,
        get_loc_from_boxes(x)[0] // HORIZONTAL_DISTANCE_RANGE))
        #bounding_boxes = [get_box_from_points(o["points"]) for o in object_list]
        bounding_boxes = [convert_box_repr(o) for o in object_list]
        locations = [((bb[0][0] + bb[0][1])/2,(bb[1][0] + bb[1][1])/2) for bb in bounding_boxes]
        list_of_nodes = []
        # Here we assume that the order is the same as the construction order
        # Adding to the list of nodes starting from the ground
        node_context = annotation["text"]
        segments = annotation["segment"]
        curr_segment = 0
        for i, bb in enumerate(bounding_boxes):
            # if o["label"] == "ground":
            #     # ground node
            #     node = SceneGraphNode.from_dict(
            #         {
            #             "node_id": 0,
            #             "block_type": "ground",
            #             "higher_level_concept": "ground",
            #             "node_context":
            #             "positional_relations": [],
            #             "bounding_box": bb
            #         }
            #     )
            #     list_of_nodes.append(node)
            # else:
            #     assert len(list_of_nodes) > 0, "Ground must be added before any node is added to the scene!"
            positional_relations = []
            # check to construct relationships with previous nodes

            for j, n in enumerate(list_of_nodes):
                prev_bb = n.bounding_box
                x_overlapped = check_overlap(prev_bb[0], bb[0])
                y_overlapped = check_overlap(prev_bb[1], bb[1])
                if x_overlapped:
                    # check for on top of/ at the bottom of relationship
                    if (prev_bb[1][0] - bb[1][1] <= VERTICAL_DISTANCE_RANGE) and (prev_bb[1][0] - bb[1][1] >= 0):
                        positional_relations.append(
                            {
                                "node_id": j,
                                "relationship": "on_the_top_of",
                            }
                        )
                    elif (bb[1][0] - prev_bb[1][1] <= VERTICAL_DISTANCE_RANGE) and (bb[1][0] - prev_bb[1][1] >= 0):
                        # This just means that i is on top of j
                        positional_relations.append(
                            {
                                "node_id": j,
                                "relationship": "on_the_bottom_of",
                            }
                        )
                elif y_overlapped:
                    # check for to the left of/ to the right of relationship
                    if (bb[0][0] - prev_bb[0][1] <= HORIZONTAL_DISTANCE_RANGE) and (bb[0][0] - prev_bb[0][1] >= 0):
                        # This just means that i is on top of j
                        positional_relations.append(
                            {
                                "node_id": j,
                                "relationship": "to_the_right_of",
                            }
                        )
                    elif (prev_bb[0][0] - bb[0][1] <= HORIZONTAL_DISTANCE_RANGE) and (prev_bb[0][0] - bb[0][1] >= 0):
                        positional_relations.append(
                            {
                                "node_id": j,
                                "relationship": "to_the_left_of",
                            }
                        )
                else:
                    if (prev_bb[1][0] - bb[1][1] <= VERTICAL_DISTANCE_RANGE) and (prev_bb[1][0] - bb[1][1] >= 0):
                        if (bb[0][0] - prev_bb[0][1] <= HORIZONTAL_DISTANCE_RANGE) and (bb[0][0] - prev_bb[0][1] >= 0):
                            positional_relations.append(
                                {
                                    "node_id": j,
                                    "relationship": "to_the_top_right_of",
                                }
                            )
                        elif (prev_bb[0][0] - bb[0][1] <= HORIZONTAL_DISTANCE_RANGE) and (prev_bb[0][0] - bb[0][1] >= 0):
                            positional_relations.append(
                                {
                                    "node_id": j,
                                    "relationship": "to_the_top_left_of",
                                }
                            )
                    elif (bb[1][0] - prev_bb[1][1] <= VERTICAL_DISTANCE_RANGE) and (bb[1][0] - prev_bb[1][1] >= 0):
                        if (bb[0][0] - prev_bb[0][1] <= HORIZONTAL_DISTANCE_RANGE) and (bb[0][0] - prev_bb[0][1] >= 0):
                            positional_relations.append(
                                {
                                    "node_id": j,
                                    "relationship": "to_the_bottom_right_of",
                                }
                            )
                        elif (prev_bb[0][0] - bb[0][1] <= HORIZONTAL_DISTANCE_RANGE) and (prev_bb[0][0] - bb[0][1] >= 0):
                            positional_relations.append(
                                {
                                    "node_id": j,
                                    "relationship": "to_the_bottom_left_of",
                                }
                            )
            if i > segments[curr_segment]:
                curr_segment += 1
            curr_context = node_context[curr_segment]
            node_dict = {
                    "node_id": i,
                    "block_type": [],
                    "higher_level_concept": None,
                    "node_context": curr_context,
                    "positional_relations": positional_relations,
                    "bounding_box": bb
            }

            node = SceneGraphNode.from_dict(node_dict)

            list_of_nodes.append(node)

        scene_graph = SceneGraph(
            name="house"
        )
        scene_graph.nodes = list_of_nodes
        return scene_graph

# Test the class
if __name__ == "__main__":
    annotation_path = "/home/local/ASUAD/weiweigu/Downloads/sample_scene.json"
    with open(annotation_path, "r") as f:
        annotation = json.load(f)
    x = SceneGraph.from_annotated_scene(annotation)
