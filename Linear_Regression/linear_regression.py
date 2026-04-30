#Preston Loera, 1001889535

import numpy as np

def mostlikelyWeights(training_file, degree, lambda1):
    phiMatrix = []
    tMatrix = []

    #Get the number of dimensions for a single input by reading "columns" of the file minus the target, reset file pointer to the beginning
    numberOfDimensions = len(training_file.readline().split()) - 1
    training_file.seek(0)
    
    numberOfInputs = 0

    #For all the inputs we must get the phi function outputs. 
    #https://athitsos.utasites.cloud/courses/cse4309_fall2025/assignments/assignment3/ is where the functions of phi are specified
    #depends on the degree set
    for line in training_file.readlines():
        temp = line.split()

        #Add target value to target matrix
        tMatrix.append(temp[-1:])
        
        #Add first element of the phi matrix
        phiMatrix.append(1.0)

        #Add rest of elements for phi
        for xinput in temp[:-1]:
            for times in range(degree):
                phiMatrix.append(np.pow(float(xinput), times + 1))

        numberOfInputs += 1

    #Convert lists into np.arrays
    phiMatrix = np.array(phiMatrix)
    tMatrix = np.array(tMatrix, dtype=float)

    #Reshape phi matrix to be a MxN matrix (M = number of inputs read from file, N = (degree * numberOfDimensions) + 1), degree specified by the number of dimensions of our data plus 1 for the first element of phi)
    phiMatrix = np.reshape(phiMatrix, (numberOfInputs, (degree * numberOfDimensions) + 1))

    #Create an identity matix of size (degree * numberOfDimensions) + 1)
    identityMatrix = np.identity((degree * numberOfDimensions) + 1)

    #Get our most likely weights from regularization slide
    weightsMostLiekly = np.linalg.pinv((identityMatrix * lambda1) + (phiMatrix.T @ phiMatrix)) @ phiMatrix.T @ tMatrix

    for index, weight in enumerate(weightsMostLiekly):
        print("w%d=%.4f" % (index, weight))

    return weightsMostLiekly

def testStage(test_file, weights, degree):
    #squaredError = 0

    #Loop through all lines in test file
    for index, line in enumerate(test_file.readlines()):
        temp = line.split()
        targetValue = line.split()[-1:]

        phiMatrix = []
        phiMatrix.append(1.0)

        #Get phi column vector
        for xinput in temp[:-1]:
            for times in range(degree):
                phiMatrix.append(np.pow(float(xinput), times + 1))   

        phiMatrix = np.array(phiMatrix, dtype=float)

        #Matrix multiply from slide 6
        output = weights.T @ phiMatrix

        print("ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f" % (index + 1, output, float(targetValue[0]), np.pow((float(targetValue[0]) - output), 2)))
        #squaredError += np.pow((float(targetValue[0]) - output), 2)

    #print(squaredError) 

def linear_regression(training_file_path, test_file_path, degree, lambda1):
    trainingFile = open(training_file_path)
    testFile = open(test_file_path)

    mostlikelyWeightsVector = mostlikelyWeights(trainingFile, degree, lambda1)
    testStage(testFile, mostlikelyWeightsVector, degree)

