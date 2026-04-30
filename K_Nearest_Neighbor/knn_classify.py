#Preston Loera, 1001889535
import numpy as np

label_to_int = {}
int_to_label = {}

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

    inputs = np.array(file_data, dtype= np.float64)
    inputs = np.reshape(inputs, (numberOfInputs, numberDimensions))

    input_labels = np.array(file_target_data, dtype=int)
    input_labels = np.reshape(input_labels, (1, numberOfInputs))

    file.close()

    return (inputs, input_labels)

def knn_classify(training_file, test_file, k):
    global label_to_int
    global int_to_label

    (training_inputs, training_labels) = readUCI(training_file, label_to_int, int_to_label)
    (test_inputs, test_labels) = readUCI(test_file, label_to_int, int_to_label)

    means = np.mean(training_inputs, axis = 0)
    stds = np.std(training_inputs, axis = 0)

    for index, std in enumerate(stds):
        if(std == 0.0):
            stds[index] = 1.0
    
    training_inputs_normalized = (training_inputs - means) / stds
    test_inputs_normalized = (test_inputs - means) / stds
    
    #Classify data
    accuracies_sum = 0.0
    for test_input_index, test_input in enumerate(test_inputs_normalized):
        closest_indexs = []
        
        distances = (test_input - training_inputs_normalized) ** 2 
        distances = np.sqrt(np.sum(distances, axis=1))

        for closest_distance_index, distance in enumerate(distances):  
            closest_indexs.append([distance, training_labels[0][closest_distance_index]])  
    
        closest_indexs = sorted(closest_indexs)
        class_distribution = np.zeros((1,len(int_to_label)), dtype=int)
        closest_indexs = closest_indexs[:k]

        for closest_index in closest_indexs:
                class_distribution[0][closest_index[1]] += 1

        class_distribution = np.array(class_distribution).flatten()
        predicted_class = np.argmax(class_distribution)
        (indices,) = np.nonzero(class_distribution == class_distribution[predicted_class])
        number_of_ties = np.prod(indices.shape)

        bypass = False
        if(number_of_ties > 1):
            for closest_index in closest_indexs:
                if(closest_index[1] == test_labels[0][test_input_index]):
                    bypass = True

        if(predicted_class == test_labels[0][test_input_index] or bypass):
            accuracy = 1.0 / number_of_ties
        else:
            accuracy = 0

        accuracies_sum += accuracy

        print('ID=%5d, predicted=%3s, true=%3s, accuracy=%4.2f' % (test_input_index + 1, int_to_label[predicted_class], int_to_label[test_labels[0][test_input_index]], accuracy))

    print('classification accuracy=%6.4f' % (accuracies_sum / len(test_inputs_normalized)))