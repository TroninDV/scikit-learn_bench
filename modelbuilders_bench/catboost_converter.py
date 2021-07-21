import daal4py
import catboost
import json
import catboost

def get_gbt_model_from_catboost(model: catboost.CatBoost):
    if not model.is_fitted():
        raise RuntimeError(
            "Model should be fitted before exporting to daal4py.")

    dump_filename = 'testing.json'
    model.save_model(dump_filename, 'json')

    with open(dump_filename) as file:
        model_data = json.load(file)

    if 'categorical_features' in model_data['features_info']:
        raise NotImplementedError(
            "Categorical features are not supported in daal4py Gradient Boosting Trees")

    n_features = len(model_data['features_info']['float_features'])
    
    if model_data['model_info']['params']['tree_learner_options']['grow_policy'] == 'SymmetricTree':
        is_symmetric_tree = True
    else:
        is_symmetric_tree = False

    if is_symmetric_tree:
        n_iterations = len(model_data['oblivious_trees'])
    else:
         n_iterations = len(model_data['trees'])

    n_classes = 0

    if 'class_params' in model_data['model_info']:
        is_classification = True
        n_classes = len(model_data['model_info']['class_params']['class_to_label'])
        mb = daal4py.gbt_clf_model_builder(n_features=n_features, n_iterations=n_iterations, n_classes=n_classes)
    else:
        is_classification = False
        mb = daal4py.gbt_reg_model_builder(n_features, n_iterations)

    splits = []

    for feature in model_data['features_info']['float_features']:
        # if feature['has_nans']:
        #     raise TypeError("Missing values are not supported in daal4py Gradient Boosting Trees")
        if feature['borders']:
            for feature_border in feature['borders']:
                splits.append({'feature_index': feature['feature_index'],
                               'value': feature_border})

    # TODO: Bias for classification???
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
                cur_tree_depth = len(
                    model_data['oblivious_trees'][tree_num]['splits'])
            else:
                cur_tree_depth = 0
            n_nodes = 2**(cur_tree_depth + 1) - 1

            if is_classification and n_classes > 2:
                tree_symmetric.append((model_data['oblivious_trees'][tree_num], cur_tree_depth))
            else:
                if not is_classification:
                    cur_tree_id = mb.create_tree(n_nodes)
                elif n_classes == 2:
                    cur_tree_id = mb.create_tree(n_nodes, 0)

                build_symmetric_tree(model_data['oblivious_trees'][tree_num], cur_tree_depth, mb, cur_tree_id, splits, is_classification, n_classes, scale, bias)
        # Depthwise
        else:
            class Node:
                def __init__(self, parent=None, split=None, value=None) -> None:
                    self.right = None
                    self.left = None
                    self.split = split
                    self.value = value

            n_nodes = 1
            if 'split' in model_data['trees'][tree_num]:
                nodes_queue = []
                root_node = Node(
                    split=splits[model_data['trees'][tree_num]['split']['split_index']])
                nodes_queue.append((model_data['trees'][tree_num], root_node))
                while nodes_queue:
                    cur_node_data, cur_node = nodes_queue.pop(0)
                    if 'value' in cur_node_data:     
                        if isinstance(cur_node_data['value'], list):  
                            cur_node.value = [value for value in cur_node_data['value']]
                        else:
                            cur_node.value = [cur_node_data['value'] * scale + bias]
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
                if isinstance(model_data['trees'][tree_num]['value'], list): 
                    root_node.value = [value * scale for value in model_data['trees'][tree_num]['value']]
                else:
                    root_node.value = [model_data['trees'][tree_num]['value'] * scale + bias]
                    
            if is_classification and n_classes > 2:
                trees_explicit.append((root_node, n_nodes))
            else:
                if not is_classification:
                    cur_tree_id = mb.create_tree(n_nodes)
                elif n_classes == 2:
                    cur_tree_id = mb.create_tree(n_nodes, 0)

                build_explicit_tree(root_node, mb, cur_tree_id)


    if is_classification and n_classes > 2:
        tree_id = []
        class_label = 0
        count = 0
        for i in range(n_iterations):
            for c in range(n_classes):     
                if is_symmetric_tree:
                    tree_id.append(mb.create_tree(2**(tree_symmetric[i][1] + 1) - 1, class_label))
                else:  
                    tree_id.append(mb.create_tree(trees_explicit[i][1], class_label))
                count += 1
                if count == n_iterations:
                    class_label += 1
                    count = 0
        if is_symmetric_tree:
            for class_label in range(n_classes):
                    for i in range(n_iterations): 
                        build_symmetric_tree(tree_symmetric[i][0],  tree_symmetric[i][1], mb, tree_id[i * n_classes + class_label], splits, 
                        is_classification, n_classes, scale, bias, class_label=class_label)
        else:
            for class_label in range(n_classes):
                for i in range(n_iterations):           
                    build_explicit_tree(trees_explicit[i][0], mb, tree_id[i * n_classes + class_label], class_label)

    return mb.model()

def build_explicit_tree(root_node, mb, cur_tree_id, class_label = 0):
    if root_node.value is None:
        root_id = mb.add_split(
            tree_id=cur_tree_id, feature_index=root_node.split['feature_index'], feature_value=root_node.split['value'])
        nodes_queue = [(root_node, root_id)]
        while nodes_queue:
            cur_node, cur_node_id = nodes_queue.pop(0)
            left_node = cur_node.left
            if left_node.value is None:
                left_node_id = mb.add_split(
                    tree_id=cur_tree_id, parent_id=cur_node_id, position=0, feature_index=left_node.split['feature_index'], 
                    feature_value=left_node.split['value'])
                nodes_queue.append((left_node, left_node_id))
            else:
                mb.add_leaf(
                tree_id=cur_tree_id, response=left_node.value[class_label], 
                parent_id=cur_node_id, position=0)
            right_node = cur_node.right
            if right_node.value is None:
                right_node_id = mb.add_split(
                    tree_id=cur_tree_id, parent_id=cur_node_id, position=1, feature_index=right_node.split['feature_index'], 
                    feature_value=right_node.split['value'])
                nodes_queue.append((right_node, right_node_id))
            else:
                mb.add_leaf(
                tree_id=cur_tree_id, response=cur_node.right.value[class_label], 
                parent_id=cur_node_id, position=1)

    else:
        mb.add_leaf(tree_id=cur_tree_id, response=root_node.value[class_label])

def build_symmetric_tree(tree_info, cur_tree_depth, mb, cur_tree_id, splits, is_classification, n_classes, scale, bias, class_label=None):
    current_tree_leaf_val = tree_info['leaf_values']
    if cur_tree_depth == 0:
        mb.add_leaf(
            tree_id=cur_tree_id, response=current_tree_leaf_val[0])
    else:
        cur_level_split = splits[tree_info['splits'][cur_tree_depth - 1]['split_index']]
        root_id = mb.add_split(
            tree_id=cur_tree_id, feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'])
        prev_level_nodes = [root_id]
        for cur_level in range(cur_tree_depth - 2, -1, -1):
            # TODO: fix variables
            cur_level_nodes = []
            for cur_parent in prev_level_nodes:
                cur_level_split = splits[tree_info['splits'][cur_level]['split_index']]
                cur_left_node = mb.add_split(
                    tree_id=cur_tree_id, parent_id=cur_parent, position=0, feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'])
                cur_right_node = mb.add_split(
                    tree_id=cur_tree_id, parent_id=cur_parent, position=1, feature_index=cur_level_split['feature_index'], feature_value=cur_level_split['value'])
                cur_level_nodes.append(cur_left_node)
                cur_level_nodes.append(cur_right_node)
            prev_level_nodes = cur_level_nodes
        if not is_classification or n_classes == 2:
            for last_level_node_num in range(len(prev_level_nodes)):            
                mb.add_leaf(
                    tree_id=cur_tree_id, response=current_tree_leaf_val[2 * last_level_node_num] * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=0)
                mb.add_leaf(
                    tree_id=cur_tree_id, response=current_tree_leaf_val[2 * last_level_node_num + 1] * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=1)
        else:      
            for last_level_node_num in range(len(prev_level_nodes)):       
                left_index = 2 * last_level_node_num * n_classes  + class_label
                right_index = (2 * last_level_node_num + 1) * n_classes  + class_label 
                mb.add_leaf(
                    tree_id=cur_tree_id, response=current_tree_leaf_val[left_index] * scale + bias , parent_id=prev_level_nodes[last_level_node_num], position=0)
                mb.add_leaf(
                    tree_id=cur_tree_id, response=current_tree_leaf_val[right_index] * scale + bias, parent_id=prev_level_nodes[last_level_node_num], position=1)
