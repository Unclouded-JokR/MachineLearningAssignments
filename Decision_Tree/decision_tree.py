#Preston Loera, 1001889535

import numpy as np
import random
import math

label_to_int = {}
int_to_label = {}

#change the name of this to something else, make children be leafs or roots, is nodes the best name for them?
class Tree:
    def __init__(self, id, nodes):
        self.tree_id = id
        self.nodes = nodes

    def print_tree(self):
        queue = []

        queue.append(self.nodes)
        while (queue):
            node = queue.pop(0)
            if(node.left_child != None and node.right_child != None):
                print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % (self.tree_id, node.node_id, node.best_attribute + 1, node.best_thr, node.gain))
                #file.write('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n' % (self.tree_id, node.node_id, node.best_attribute + 1, node.best_thr, node.gain))
            else:
                print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % (self.tree_id, node.node_id, node.best_attribute, node.best_thr, node.gain))
                #file.write('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n' % (self.tree_id, node.node_id, node.best_attribute, node.best_thr, node.gain))

            #Add children if the node has any
            if(node.left_child != None):
                queue.append(node.left_child)
            
            if(node.right_child != None):
                queue.append(node.right_child)

    def evaluate_mostlikely(self, example):
        node = self.nodes
        distribution_to_return = None

        while(True):
            best_attribute = node.best_attribute
            best_thr = node.best_thr

            #Know we are at a leaf if both children are None
            if(node.left_child == None and node.right_child == None):
                distribution_to_return = node.node_distribution
                break

            if(example[best_attribute] < best_thr):
                node = node.left_child
            else:
                node = node.right_child

        return distribution_to_return

class Node:
    def __init__(self, best_attribute, best_thr, gain, node_id):
        self.best_attribute = best_attribute
        self.best_thr = best_thr
        self.gain = gain
        self.node_id = node_id

class Root(Node):
    def __init__(self, best_attribute, best_thr, gain, node_id):
        super().__init__(best_attribute, best_thr, gain, node_id)    
        self.left_child = None
        self.right_child = None

class Leaf(Node):
    def __init__(self, node_id, distribution):
        super().__init__(-1, -1, 0, node_id)
        self.node_distribution = distribution
        self.left_child = None
        self.right_child = None

def distribution(examples):
    global label_to_int
    distribution = np.zeros((1, len(label_to_int)), dtype=float)

    numLabels = 0
    for exampleLabel in examples:
        distribution[0][exampleLabel] += 1
        numLabels += 1

    return distribution / numLabels

def check_same_class(examples):
    firstLabel = examples[1][0][0]

    for exampleLabel in examples[1][0]:
        if(exampleLabel != firstLabel):
            return False
        
    return True

def min(attribute_values):
    min = np.inf

    for attribute_value in attribute_values:
        if(attribute_value < min):
            min = attribute_value

    return min

def max(attribute_values):
    max = -np.inf

    for attribute_value in attribute_values:
        if(attribute_value > max):
            max = attribute_value

    return max

def entropy(examples_labels, total_examples): 
    if(total_examples == 0):
        return 0

    global label_to_int 
    distribution = np.bincount(examples_labels, minlength = len(label_to_int))
    distribution_zero_counts = np.where(distribution == 0)

    for zero_index in distribution_zero_counts:
        distribution[zero_index] = total_examples

    distribution = distribution / total_examples

    entropy = -np.sum(distribution * np.log2(distribution), dtype=float)
        
    return entropy
    
def information_gain(examples, attribute, threshold): 
    total_examples = examples[0].shape[0] 
    example_data, example_labels = examples[0], examples[1][0]

    examples_below_thresh = []
    examples_above_thresh = []

    for index, example_attribute in enumerate(example_data[:,attribute]):
        if(example_attribute < threshold):
            examples_below_thresh.append(example_labels[index])
        else:
            examples_above_thresh.append(example_labels[index])

    examples_below_thresh = np.array(examples_below_thresh, dtype=int)
    examples_above_thresh = np.array(examples_above_thresh, dtype=int)
    below_thresh = examples_below_thresh.shape[0]
    above_thresh = examples_above_thresh.shape[0]
    
    gain = (
        entropy(examples[1][0], total_examples) 
        - ((below_thresh / total_examples) * entropy(examples_below_thresh, below_thresh)) 
        - ((above_thresh / total_examples) * entropy(examples_above_thresh, above_thresh))
    )

    return gain

def examples_below_thr(examples, best_attribute, threshold):
    examples_below = []
    example_labels = []
    num_inputs = 0
    for index, example in enumerate(examples[0]):
        
        if(example[best_attribute] < threshold):
            num_inputs += 1
            examples_below.extend(examples[0][index])
            example_labels.append(examples[1][0][index])

    inputs = np.array(examples_below, dtype=float)
    inputs = np.reshape(inputs, (num_inputs, examples[0].shape[1]))

    input_labels = np.array(example_labels, dtype=int)
    input_labels = np.reshape(input_labels, (1, num_inputs))

    return (inputs, input_labels)

def examples_above_thr(examples, best_attribute, threshold):
    examples_below = []
    example_labels = []
    num_inputs = 0
    for index, example in enumerate(examples[0]):
        
        if(example[best_attribute] >= threshold):
            num_inputs += 1
            examples_below.extend(examples[0][index])
            example_labels.append(examples[1][0][index])

    inputs = np.array(examples_below, dtype=float)
    inputs = np.reshape(inputs, (num_inputs, examples[0].shape[1]))

    input_labels = np.array(example_labels, dtype=int)
    input_labels = np.reshape(input_labels, (1, num_inputs))

    return (inputs, input_labels)

def choose_attribute(examples, attributes):
    max_gain = best_threshhold = best_attribute = -1

    for attribute in range(attributes):
        attribute_values = examples[0][:,attribute]
        L = min(attribute_values)
        M = max(attribute_values)

        for K in range(1, 51):
            threshold = L + ((K*(M-L))/51)
            gain = information_gain(examples, attribute, threshold)

            if(gain > max_gain):
                max_gain = gain
                best_attribute = attribute
                best_threshhold = threshold

    return (best_attribute, best_threshhold, max_gain)

def choose_attribute_random(examples, attributes):
    max_gain = best_threshhold = -1
    A = random.randint(0, attributes - 1)

    attribute_values = examples[0][:,A]
    L = min(attribute_values)
    M = max(attribute_values)

    for K in range(1, 51):
        threshold = L + ((K*(M-L))/51)
        gain = information_gain(examples, A, threshold)

        if(gain > max_gain):
            max_gain = gain
            best_threshhold = threshold

    return (A, best_threshhold, max_gain)

#Optimized version of decision tree learning
def decision_tree_learning(examples, attributes, default, pruning_thr, node_id):
    global label_to_int

    if(len(examples[0]) < pruning_thr):
        return Leaf(node_id, default)
    elif(check_same_class(examples)):
        #return label of the class as a distribution to make consistent with other leaves
        node_distribution = np.zeros((1, len(label_to_int)), dtype=float)
        node_distribution[0][examples[1][0]] = 1.0
        return Leaf(node_id, node_distribution)
    else:
        (best_attribute, best_thr, gain) = choose_attribute(examples, attributes)
        root = Root(best_attribute, best_thr, gain, node_id)

        examples_left = examples_below_thr(examples, best_attribute, best_thr)
        examples_right = examples_above_thr(examples, best_attribute, best_thr)

        root.left_child = decision_tree_learning(examples_left, attributes, distribution(examples[1][0]), pruning_thr, 2 * node_id)
        root.right_child = decision_tree_learning(examples_right, attributes, distribution(examples[1][0]), pruning_thr, (2 * node_id) + 1)

        return root

#Randomized version of decision tree learning 
def decision_tree_learning_randomized(examples, attributes, default, pruning_thr, node_id):
    global label_to_int

    if(len(examples[0]) < pruning_thr):
        return Leaf(node_id, default)
    elif(check_same_class(examples)):
        #return label of the class as a distribution to make consistent with other leaves
        node_distribution = np.zeros((1, len(label_to_int)), dtype=float)
        node_distribution[0][examples[1][0]] = 1.0
        return Leaf(node_id, node_distribution)
    else:
        (best_attribute, best_thr, gain) = choose_attribute_random(examples, attributes)
        root = Root(best_attribute, best_thr, gain, node_id)

        examples_left = examples_below_thr(examples, best_attribute, best_thr)
        examples_right = examples_above_thr(examples, best_attribute, best_thr)

        root.left_child = decision_tree_learning_randomized(examples_left, attributes, distribution(examples[1][0]), pruning_thr, 2 * node_id)
        root.right_child = decision_tree_learning_randomized(examples_right, attributes, distribution(examples[1][0]), pruning_thr, (2 * node_id) + 1)

        return root

def decision_tree_learning_top_level(examples, pruning_thr, random_attribute_bool, tree_id):
    attributes = examples[0].shape[1]
    default = distribution(examples[1][0])
    
    if(random_attribute_bool == True):
        return Tree(tree_id, decision_tree_learning_randomized(examples, attributes, default, pruning_thr, 1))
    else:
        return Tree(tree_id, decision_tree_learning(examples, attributes, default, pruning_thr, 1))

def readUCI(file_path, label_to_int, int_to_label):
    file = open(file_path)

    file_data = []
    file_target_data = []

    numberOfInputs = 0
    numberDimensions = len(file.readline().strip().split()[:-1])
    file.seek(0)

    for fileline in file.readlines():
        data = fileline.strip().split()

        #Check if class label is in our known class labels
        if(data[-1] not in label_to_int):
            label_to_int[data[-1]] = len(label_to_int)
            int_to_label[label_to_int[data[-1]]] = data[-1]

    file.seek(0)

    for fileline in file.readlines():
        data = fileline.strip().split()[:-1]
        classlabel = fileline.strip().split()[-1]
        file_target_data.append(label_to_int[classlabel])

        for index in range(len(data)):
            data[index] = float(data[index])

        numberOfInputs += 1
        file_data.extend(data)

    inputs = np.array(file_data, dtype=float)
    inputs = np.reshape(inputs, (numberOfInputs, numberDimensions))

    input_labels = np.array(file_target_data, dtype=int)
    input_labels = np.reshape(input_labels, (1, numberOfInputs))

    file.close()

    return (inputs, input_labels)

def decision_tree(training_file, test_file, option, pruning_thr):
    global label_to_int
    global int_to_label
    # file_name = "tree_output.txt"
    # file = open(file_name, 'w')

    (training_inputs, training_labels) = readUCI(training_file, label_to_int, int_to_label)
    (test_inputs, test_labels) = readUCI(test_file, label_to_int, int_to_label)

    if(option == 'optimized'):
        tree = decision_tree_learning_top_level((training_inputs, training_labels), pruning_thr, False, 1)
        tree.print_tree()

        accuracies = []
        for index, test_input in enumerate(test_inputs):
            tree_distribution = tree.evaluate_mostlikely(test_input)

            tree_output = tree_distribution.flatten()
            predicted_class = np.argmax(tree_output)
            (indices,) = np.nonzero(tree_output == tree_output[predicted_class])
            number_of_ties = np.prod(indices.shape)

            if(predicted_class == test_labels[0][index]):
                accuracy = 1.0 / number_of_ties
            else:
                accuracy = 0

            accuracies.append(accuracy)
    else:
        trees = []
        for i in range(option):
            trees.append(decision_tree_learning_top_level((training_inputs, training_labels), pruning_thr, True, i + 1))

        for tree in trees:
            tree.print_tree()

        accuracies = []
        for index, test_input in enumerate(test_inputs):
            average_dis = []
            distributions = []
            
            for tree in trees:
                distributions.append(tree.evaluate_mostlikely(test_input))
            
            for i in range(len(distributions[0])):
                average_prob = 0.0
                for distribution in distributions:
                    average_prob += distribution[i]
                average_dis.append(average_prob / len(distributions))

            tree_output = np.array(average_dis).flatten()
            predicted_class = np.argmax(tree_output)
            (indices,) = np.nonzero(tree_output == tree_output[predicted_class])
            number_of_ties = np.prod(indices.shape)

            if(predicted_class == test_labels[0][index]):
                accuracy = 1.0 / number_of_ties
            else:
                accuracy = 0

            accuracies.append(accuracy)


    total = 0.0
    for accuracy in accuracies:
        total += accuracy

    print('classification accuracy=%6.4f' % (total / len(accuracies)))
    # file.close()
