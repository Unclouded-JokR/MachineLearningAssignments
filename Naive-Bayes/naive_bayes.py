#Preston Loera, 1001889535

import numpy as np
import random

classDistribution = np.array([])
classLabelDictionary = {}

class ClassAttribute:
    def __init__(self, attributeNumber, classNumber, numAttributes):
        self.attributeNumber = attributeNumber
        self.classNumber = classNumber
        self.numberOfAttributes = numAttributes
        self.data = []

    def calculateGuassians(self, x):
        return (1 / (self.std) * pow((2 * np.pi), .5)) * np.exp(-1 * ((pow((x - self.mean), 2)) / (2 * pow(self.std, 2))))

    def convertDataToNP(self):
        if(len(self.data) == 0):
            self.mean = 0.0
            self.std = .01
        else:
            self.data = np.array(self.data, dtype=float)
            
            self.n = len(self.data)
            self.mean = np.mean(self.data)
            self.std = np.std(self.data)

            if(self.std < .01):
                self.std = .01

    def __str__(self):
        return "Class %d, attribute %d, mean = %.2f, std = %.2f" % (self.classNumber, self.attributeNumber, self.mean, self.std)



def createArray(file):
    global classLabelDictionary

    uniqueClassLabels = []
    dimensions = len(file.readline().split()) - 1 #number of attributes for every x

    #determine the largest class identifier for the shape of array
    for fileline in file.readlines():
        data = fileline.split()

        if(int(data[int(len(data) - 1)]) not in uniqueClassLabels):
            uniqueClassLabels.append(int(data[int(len(data) - 1)]))

    #create array of size largestClass x dimensions and initialize all values
    array = np.zeros([len(uniqueClassLabels), dimensions], dtype=ClassAttribute)
    classes, attributes = array.shape
    uniqueClassLabels.sort()

    #Create a dictionary to make cases of classes starting from 0 easier to deal with
    
    for i in range(len(uniqueClassLabels)):
        classLabelDictionary[str(uniqueClassLabels[i])] = i

    #Class number is the rows and attributes is the columns
    for i, classnumber in enumerate(uniqueClassLabels):
        for attribute in range(attributes):
            array[i, attribute] = ClassAttribute(attribute + 1,classnumber, attributes)

    return array
    

def readTrainingContents(dataArray, trainingFile):
    global classDistribution
    global classLabelDictionary

    #Create np.array to get the distribution of each class label in the training file
    classDistribution = np.zeros([dataArray.shape[0]], dtype=int)

    #For loop through file to calculate probability of C for each class label and to add data to each Guassian
    for fileline in trainingFile.readlines():
        
        #Get the string version of the class label to get the proper index from the class label dicitionay
        classIdString = fileline.split()[dataArray.shape[1]]
        classId = classLabelDictionary[classIdString]

        #increment class label frequncy
        classDistribution[classId] += 1

        #append data to each guassian for that file line
        for attribute, element in enumerate(fileline.split()):
             if(attribute < int(len(fileline.split()) - 1)):
                 dataArray[classId, attribute].data.append(element)
             else:
                continue

    rows, columns = dataArray.shape

    for row in range(rows):
        for column in range(columns):
            dataArray[row, column].convertDataToNP()
            print(dataArray[row, column])



def calculateClassProb():
    global classDistribution

    classProbabiltiy = np.empty([classDistribution.shape[0]], dtype=float)
    totalTestObjects = 0

    for i in range(len(classDistribution)):
        totalTestObjects += classDistribution[i]

    for i in range(len(classDistribution)):
        classProbabiltiy[i] = classDistribution[i] / totalTestObjects
    
    classDistribution = classProbabiltiy     

def argmax(testFile, guassianDataSet):
    global classDistribution
    global classLabelDictionary

    #Store all the accuracies to compute total accuracy of model at the end
    accuracies = []

    for id, fileline in enumerate(testFile.readlines()):

        #split file line and exact all columns except the last one
        fileline = fileline.split()
        xTraining = np.array(fileline[:-1],dtype=float)

        #create a new classifier of size numberofclasses in data set for each new X vector
        classifier = np.empty([classDistribution.shape[0]], dtype=float)

        classes, attributes = guassianDataSet.shape

        #Calculate p(x|Ck) for every class Ck and appened to classifier np array
        for classnumber in range(classes):

            #Every time we check a new class make sure to set the variables back to their original value
            pXCk = 1
            pX = 0

            #Get every p(X | Ck) since x is a multi dimension variable
            for attribute in range(attributes):
                pXCk *= guassianDataSet[classnumber,attribute].calculateGuassians(xTraining[attribute])

            #Numerator of p(Ck | x) 
            numerator = pXCk * classDistribution[classnumber]

            #Calculate p(x), so find p(X | Ck) for all attributes of X for a class and add them to the sum pxck (pxck is similar to the variable pXCk since we are doing the same operation just for ALL classes) 
            for classnumber2 in range(classes):
                pxck = 1.0

                for attribute2 in range(attributes):
                    pxck *= guassianDataSet[classnumber2,attribute2].calculateGuassians(xTraining[attribute2])

                pX += pxck * classDistribution[classnumber2]

            classifier[classnumber] = numerator / pX

        #Used for debugging
        # total = 0.0
        # for number in classifier:
        #     total += number
        # print(total)

        #selected index stores the index of the class we are doing to choose
        #likely hood is how sure we are for selected that class index
        #accuracy is how correct our naivebayes choose 
        selectedIndex = []
        selectedLikelyhood = -9999.0
        accuracy = 0.0

        #Create a list of keys that the index is able to retrieve based on most likely element
        labels = list(classLabelDictionary.keys())

        #loop through all classes in classifier, and check all the probabilities for the most likely one, append to list if we have a tie
        for index, classProbability in enumerate(classifier):
            if(classProbability > selectedLikelyhood):
                selectedIndex.clear()
                selectedIndex.append(labels[index])
                selectedLikelyhood = classProbability
            elif((classProbability - selectedLikelyhood) >= .05):
                selectedIndex.append(labels[index])

        #true case, is for ties, flase case, is for a single element in selectedIndex
        if(len(selectedIndex) > 1):
            selectedIndex = random.sample(selectedIndex, 1)

            if(selectedIndex[0] == fileline[-1]):
                accuracy = 1 / len(selectedIndex)
            else:
                accuracy = 0.0
        else:
            if(selectedIndex[0] == fileline[-1]):
                accuracy = 1.0
            else:
                accuracy = 0.0

        print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n" % (id + 1, int(selectedIndex[0]), selectedLikelyhood, int(fileline[-1]), accuracy), end="")
        accuracies.append(accuracy)

    total = 0.0
    for accuracy in accuracies:
        total += accuracy

    print("classification accuracy=%6.4f" % (total / len(accuracies)))



def naive_bayes(trainingFileDir, testFileDir):
    trainingFile = open(trainingFileDir)
    testFile = open(testFileDir)

    #Create array of size class identifiers x number of attributes
    array = createArray(trainingFile)
    trainingFile.seek(0)
    readTrainingContents(array, trainingFile)

    #Update class probabilities
    calculateClassProb()

    argmax(testFile, array)
