#Preston Loera, 1001889535
import numpy as np
import tensorflow as tf
import random

stringLabelDictionary = None

def readUCITraining(trainingData):
    global stringLabelDictionary
    
    numberOfInputs = 0
    targetStringLables = dict()
    normalizedData = []
    mappedClassLabels = []

    numberDimensions = len(trainingData.readline().strip().split()[:-1])
    trainingData.seek(0)

    #Find unique class labels and add them to dictionary
    for fileline in trainingData.readlines():
        data = fileline.strip().split()
        if(data[-1] not in targetStringLables):
            targetStringLables[data[-1]] = len(targetStringLables)

    trainingData.seek(0)

    for fileline in trainingData.readlines():
        data = fileline.strip().split()[:-1]
        classlabel = fileline.strip().split()[-1]
        mappedClassLabels.append(targetStringLables[classlabel])

        for index in range(len(data)):
            data[index] = float(data[index])

        numberOfInputs += 1
        normalizedData.extend(data)

    #Convert and reshape the normalized data and class labels to into a (Number of inputs X dimensions) array and a (number of inputs X 1) array
    normalizedData = np.array(normalizedData, dtype=float)
    normalizedData = np.reshape(normalizedData, (numberOfInputs, numberDimensions))

    mappedClassLabels = np.array(mappedClassLabels, dtype=int)
    mappedClassLabels = np.reshape(mappedClassLabels, (numberOfInputs, 1))
    
    stringLabelDictionary = targetStringLables
    trainingData.close()

    return normalizedData, mappedClassLabels

def readUCITest(testingData):
    global stringLabelDictionary
    
    numberOfInputs = 0
    normalizedData = []
    mappedClassLabels = []

    numberDimensions = len(testingData.readline().strip().split()[:-1])
    testingData.seek(0)

    for fileline in testingData.readlines():
        data = fileline.strip().split()[:-1]
        classlabel = fileline.strip().split()[-1]
        mappedClassLabels.append(stringLabelDictionary[classlabel])

        for index in range(len(data)):
            data[index] = float(data[index])

        numberOfInputs += 1
        normalizedData.extend(data)

    #Convert and reshape the normalized data and class labels to into a (Number of inputs X dimensions) array and a (number of inputs X 1) array
    normalizedData = np.array(normalizedData, dtype=float)
    normalizedData = np.reshape(normalizedData, (numberOfInputs, numberDimensions))

    mappedClassLabels = np.array(mappedClassLabels, dtype=int)
    mappedClassLabels = np.reshape(mappedClassLabels, (numberOfInputs, 1))
    testingData.close()

    return normalizedData, mappedClassLabels

def nn_keras(directory, dataset, numLayers, units_per_layer, numEpochs):
    global stringLabelDictionary

    trainingFile = directory + '/' + dataset + '_training.txt'
    testFile = directory + '/' + dataset + '_test.txt'
    (training_inputs, training_labels) = readUCITraining(open(trainingFile))
    (test_inputs, test_labels) = readUCITest(open(testFile))
    max_value = np.max(np.abs(training_inputs))
    training_inputs = training_inputs / max_value
    test_inputs = test_inputs/ max_value

    # print(training_inputs.shape)
    # print(training_labels.shape)
    # print(test_inputs.shape)
    # print(test_labels.shape)

    input_shape = training_inputs[0].shape
    print(input_shape)
    number_of_classes = np.max([np.max(training_labels), np.max(test_labels)]) + 1

    if(numLayers < 2):
        print('Invalid number of layers, please enter value >= 2.')
        return

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape = input_shape))
    numLayers -= 1

    while(numLayers != 1):
        model.add(tf.keras.layers.Dense(units_per_layer, activation='sigmoid'))
        numLayers -= 1

    model.add(tf.keras.layers.Dense(number_of_classes, activation='sigmoid'))

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    model.fit(training_inputs, training_labels, epochs=numEpochs)

    #Testing using tensorflow evaluate and summary
    # test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)
    # print('\nTest accuracy: %.2f%%' % (test_acc * 100))
    #print(model.summary())

    accuracies = []

    #Create a list of keys that the index is able to retrieve based on most likely element
    labels = list(stringLabelDictionary.keys())

    for object_id in range(len(test_inputs)):
        input_vector = test_inputs[object_id,:]
        true_class = test_labels[object_id][0]
        input_vector = np.reshape(input_vector, (1, training_inputs.shape[1]))
        nn_output = model.predict(input_vector).flatten()

        accuracy = 0.0

        predicted_class = np.argmax(nn_output)
        (indices,) = np.nonzero(nn_output == nn_output[predicted_class])
        number_of_ties = np.prod(indices.shape)

        if (nn_output[true_class] == nn_output[predicted_class]):
            accuracy = 1.0 / number_of_ties
        else:
            accuracy = 0

        #Evil old code that does not work for some strange reason I cannot figure out at 2:00 am
        # selectedIndex = []
        # selectedLikelyhood = 0.0

        # #loop through all classes in classifier, and check all the probabilities for the most likely one, append to list if we have a tie
        # for index, classProbability in enumerate(nn_output):
        #     if(classProbability > selectedLikelyhood):
        #         selectedIndex.clear()
        #         selectedIndex.append(labels[index])
        #         selectedLikelyhood = classProbability
        #     elif(np.abs(classProbability - selectedLikelyhood) <= .05):
        #         selectedIndex.append(labels[index])

        # #true case, is for ties, false case, is for a single element in selectedIndex
        # if(len(selectedIndex) > 1):
        #     selectRandom = random.sample(selectedIndex, 1)

        #     if(stringLabelDictionary[selectRandom[0]] == true_class):
        #         accuracy = 1 / len(selectedIndex)
        #     else:
        #         accuracy = 0.0
        # else:
        #     if(stringLabelDictionary[selectedIndex[0]] == true_class):
        #         accuracy = 1.0
        #     else:
        #         accuracy = 0.0

        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (object_id + 1, predicted_class, true_class, accuracy))
        accuracies.append(accuracy)

    total = 0.0
    for accuracy in accuracies:
        total += accuracy

    print("classification accuracy=%6.4f" % (total / len(accuracies)))
    

