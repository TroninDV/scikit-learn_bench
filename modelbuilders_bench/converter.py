from typing import List, Deque, Dict, Any
from collections import deque
from os import remove, getpid
import json
import re
from time import time
from daal4py import gbt_clf_model_builder

def get_gbt_model_from_catboost(model: Any) -> Any:
    if not model.is_fitted():
        raise RuntimeError(
            "Model should be fitted before exporting to daal4py.")

    dump_filename = f"catboost_model_{getpid()}_{time()}"

    # Dump model in file
    model.save_model(dump_filename, 'json')

    # Read json with model
    with open(dump_filename) as file:
        model_data = json.load(file)

    # Delete dump file
    remove(dump_filename)

    if 'categorical_features' in model_data['features_info']:
        raise NotImplementedError(
            "Categorical features are not supported in daal4py Gradient Boosting Trees")

    n_features = len(model_data['features_info']['float_features'])

    is_symmetric_tree = model_data['model_info']['params']['tree_learner_options']['grow_policy'] == 'SymmetricTree'

    if is_symmetric_tree:
        n_iterations = len(model_data['oblivious_trees'])
    else:
        n_iterations = len(model_data['trees'])

    n_classes = 0

    if 'class_params' in model_data['model_info']:
        is_classification = True
        n_classes = len(model_data['model_info']
                        ['class_params']['class_to_label'])
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes)
    else:
        is_classification = False
        mb = gbt_reg_model_builder(n_features, n_iterations)

    splits = []

    # Create splits array (all splits are placed sequentially)
    for feature in model_data['features_info']['float_features']:
        if feature['borders']:
            for feature_border in feature['borders']:
                splits.append(
                    {'feature_index': feature['feature_index'], 'value': feature_border})

    if not is_classification:
        bias = model_data['scale_and_bias'][1][0] / n_iterations
        scale = model_data['scale_and_bias'][0]
    else:
        bias = 0
        scale = 1

    trees_explicit = []
    tree_symmetric = []

    for tree_num in range(n_iterations):
        if is_symmetric_tree:
            
            if model_data['oblivious_trees'][tree_num]['splits'] is not None:
                # Tree has more than 1 node
                cur_tree_depth = len(
                    model_data['oblivious_trees'][tree_num]['splits'])
            else:
                cur_tree_depth = 0

            tree_symmetric.append(
                (model_data['oblivious_trees'][tree_num], cur_tree_depth))
        else:
            class Node:
                def __init__(self, parent=None, split=None, value=None) -> None:
                    self.right = None
                    self.left = None
                    self.split = split
                    self.value = value

            n_nodes = 1
            # Check if node is a leaf (in case of stump)
            if 'split' in model_data['trees'][tree_num]:
                # Get number of trees and splits info via BFS
                # Create queue
                nodes_queue = []
                root_node = Node(
                    split=splits[model_data['trees'][tree_num]['split']['split_index']])
                nodes_queue.append((model_data['trees'][tree_num], root_node))
                while nodes_queue:
                    cur_node_data, cur_node = nodes_queue.pop(0)
                    if 'value' in cur_node_data:
                        if isinstance(cur_node_data['value'], list):
                            cur_node.value = [
                                value for value in cur_node_data['value']]
                        else:
                            cur_node.value = [
                                cur_node_data['value'] * scale + bias]
                    else:
                        cur_node.split = splits[cur_node_data['split']
                                                ['split_index']]
                        left_node = Node()
                        right_node = Node()
                        cur_node.left = left_node
                        cur_node.right = right_node
                        nodes_queue.append((cur_node_data['left'], left_node))
                        nodes_queue.append(
                            (cur_node_data['right'], right_node))
                        n_nodes += 2
            else:
                root_node = Node()
                if is_classification and n_classes > 2:
                    root_node.value = [
                        value * scale for value in model_data['trees'][tree_num]['value']]
                else:
                    root_node.value = [model_data['trees'][tree_num]['value'] * scale + bias]
            trees_explicit.append((root_node, n_nodes))

    tree_id = []
    class_label = 0
    count = 0

    # Only 1 tree for each iteration in case of regression or binary classification
    if not is_classification or n_classes == 2:
        n_tree_each_iter = 1
    else:
        n_tree_each_iter = n_classes

    # Create id for trees (for the right order in modelbuilder)
    for i in range(n_iterations):
        for c in range(n_tree_each_iter):
            if is_symmetric_tree:
                n_nodes = 2**(tree_symmetric[i][1] + 1) - 1
            else:
                n_nodes = trees_explicit[i][1]

            if is_classification and n_classes > 2:
                tree_id.append(mb.create_tree(n_nodes, class_label))
                count += 1
                if count == n_iterations:
                    class_label += 1
                    count = 0

            elif is_classification:
                tree_id.append(mb.create_tree(n_nodes, 0))
            else:
                tree_id.append(mb.create_tree(n_nodes))
    

    if is_symmetric_tree:
        for class_label in range(n_tree_each_iter):
            for i in range(n_iterations):
                cur_tree_info = tree_symmetric[i][0]
                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                cur_tree_leaf_val = cur_tree_info['leaf_values']
                cur_tree_depth = tree_symmetric[i][1]

                if cur_tree_depth == 0:
                    mb.add_leaf(
                        tree_id=cur_tree_id, response=cur_tree_leaf_val[0])
                else:
                    # One split used for the whole level 
                    cur_level_split = splits[cur_tree_info['splits']
                                             [cur_tree_depth - 1]['split_index']]
                    root_id = mb.add_split(
                        tree_id=cur_tree_id, feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'])
                    prev_level_nodes = [root_id]

                    # Iterate over levels, splits in json are reversed (root split is the last)
                    for cur_level in range(cur_tree_depth - 2, -1, -1):
                        cur_level_nodes = []
                        for cur_parent in prev_level_nodes:
                            cur_level_split = splits[cur_tree_info['splits']
                                                     [cur_level]['split_index']]
                            cur_left_node = mb.add_split(tree_id=cur_tree_id, parent_id=cur_parent, position=0,
                                                         feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'])
                            cur_right_node = mb.add_split(tree_id=cur_tree_id, parent_id=cur_parent, position=1,
                                                          feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'])
                            cur_level_nodes.append(cur_left_node)
                            cur_level_nodes.append(cur_right_node)
                        prev_level_nodes = cur_level_nodes

                    # Different storing format for leaves
                    if not is_classification or n_classes == 2:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            mb.add_leaf(tree_id=cur_tree_id, response=cur_tree_leaf_val[2 * last_level_node_num]
                                        * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=0)
                            mb.add_leaf(tree_id=cur_tree_id, response=cur_tree_leaf_val[2 * last_level_node_num + 1]
                                        * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=1)
                    else:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            left_index = 2 * last_level_node_num * n_tree_each_iter + class_label
                            right_index = (2 * last_level_node_num + 1) * \
                                n_tree_each_iter + class_label
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=cur_tree_leaf_val[left_index] * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=0)
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=cur_tree_leaf_val[right_index] * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=1)
    else:
        for class_label in range(n_tree_each_iter):
            for i in range(n_iterations):
                root_node = trees_explicit[i][0]

                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                # Traverse tree via BFS and build tree with modelbuilder
                if root_node.value is None:
                    root_id = mb.add_split(
                        tree_id=cur_tree_id, feature_index=root_node.split['feature_index'], feature_value=root_node.split['value'])
                    nodes_queue = [(root_node, root_id)]
                    while nodes_queue:
                        cur_node, cur_node_id = nodes_queue.pop(0)
                        left_node = cur_node.left
                        # Check if node is a leaf
                        if left_node.value is None:
                            left_node_id = mb.add_split(tree_id=cur_tree_id, parent_id=cur_node_id, position=0,
                                                        feature_index=left_node.split['feature_index'], feature_value=left_node.split['value'])
                            nodes_queue.append((left_node, left_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=left_node.value[class_label], parent_id=cur_node_id, position=0)
                        right_node = cur_node.right
                        # Check if node is a leaf
                        if right_node.value is None:
                            right_node_id = mb.add_split(tree_id=cur_tree_id, parent_id=cur_node_id, position=1,
                                                         feature_index=right_node.split['feature_index'], feature_value=right_node.split['value'])
                            nodes_queue.append((right_node, right_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id, response=cur_node.right.value[class_label],
                                parent_id=cur_node_id, position=1)

                else:
                    # Tree has only one node
                    mb.add_leaf(tree_id=cur_tree_id,
                                response=root_node.value[class_label])

    return mb.model()