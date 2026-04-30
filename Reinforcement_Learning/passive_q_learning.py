#Preston Loera, 1001889535

from abc import ABC, abstractmethod
import numpy as np
import random
enviormentSize = [-1, -1]

class State(ABC):
    @abstractmethod
    def senseReward(self):
        pass

class TerimnalState(State):
    def __init__(self, reward):
        self.stateReward = reward
        self.type = "terminal" 
    
    def senseReward(self):
        return self.stateReward

class BlockedState(State):
    def __init__(self):
        self.stateReward = 0.0
        self.type = "blocked" 
  
    def senseReward(self):
        return self.stateReward
    
class NonTerimnalState(State):
    def __init__(self, index, reward):
        self.index = index
        self.row = self.index[0]
        self.column = self.index[1]
        self.stateReward = reward
        self.type = "non-terminal" 

    def up(self, probability, enviorment):
        global enviormentSize

        if(probability <= 80):
            #Up
            if(self.row - 1 < 0 or (enviorment[self.row - 1, self.column].type == "blocked")):
                return self
            else:
                return enviorment[self.row - 1, self.column]
        elif(probability > 80 and probability <= 90):
             #Right
             if(self.column + 1 == enviormentSize[1] or (enviorment[self.row, self.column + 1].type == "blocked")):
                return self
             else:
                return enviorment[self.row, self.column + 1]
        elif(probability > 90 and probability <= 100):
            #Left
            if(self.column - 1 < 0 or (enviorment[self.row, self.column - 1].type == "blocked")):
                 return self
            else:
                return enviorment[self.row, self.column - 1]

    def right(self, probability, enviorment):
        global enviormentSize

        if(probability <= 80):
            #Right
            if(self.column + 1 == enviormentSize[1] or (enviorment[self.row, self.column + 1].type == "blocked")):
                return self
            else:
                return enviorment[self.row, self.column + 1]
        elif(probability > 80 and probability <= 90):
            #Up
            if(self.row - 1 < 0 or (enviorment[self.row - 1, self.column].type == "blocked")):
                return self
            else:
                return enviorment[self.row - 1, self.column]
        elif(probability > 90 and probability <= 100):
            #Down
            if(self.row + 1 == enviormentSize[0] or (enviorment[self.row + 1, self.column].type == "blocked")):
                return self
            else:
                return enviorment[self.row + 1, self.column]

    def down(self, probability, enviorment):
        global enviormentSize

        if(probability <= 80):
            #Down
            if(self.row + 1 == enviormentSize[0] or (enviorment[self.row + 1, self.column].type == "blocked")):
                return self
            else:
                return enviorment[self.row + 1, self.column]
        elif(probability > 80 and probability <= 90):
            #Left
            if(self.column - 1 < 0 or (enviorment[self.row, self.column - 1].type == "blocked")):
                 return self
            else:
                return enviorment[self.row, self.column - 1]
        elif(probability > 90 and probability <= 100):
            #Right
             if(self.column + 1 == enviormentSize[1] or (enviorment[self.row, self.column + 1].type == "blocked")):
                return self
             else:
                return enviorment[self.row, self.column + 1]
        
    def left(self, probability, enviorment):
        global enviormentSize

        if(probability <= 80):
            #Left
            if(self.column - 1 < 0 or (enviorment[self.row, self.column - 1].type == "blocked")):
                 return self
            else:
                return enviorment[self.row, self.column - 1]
        elif(probability > 80 and probability <= 90):
            #Down
            if(self.row + 1 == enviormentSize[0] or (enviorment[self.row + 1, self.column].type == "blocked")):
                return self
            else:
                return enviorment[self.row + 1, self.column]
        elif(probability > 90 and probability <= 100):
            #Up
            if(self.row - 1 < 0 or (enviorment[self.row - 1, self.column].type == "blocked")):
                return self
            else:
                return enviorment[self.row - 1, self.column]

    def senseReward(self):
        return self.stateReward

    def executeAction(self, action, enviorment):
        prob = random.randint(1, 100)

        match action:
            case "<":
                return self.left(prob, enviorment)
            case ">":
                return self.right(prob, enviorment)
            case "^":
                return self.up(prob, enviorment)
            case "v":
                return self.down(prob, enviorment)

def definePolicy(policy_file):
    policy_file = open(policy_file)
    policy = np.array([])

    numColumns = len(policy_file.readline().strip().split(','))
    policy_file.seek(0)

    numRows = 0
    for row in policy_file.readlines():
        row = row.strip().split(',')
        temp = np.array([])

        for state in row:
            state = state.strip()
            temp = np.append(temp, state)

        policy = np.append(policy, temp)
        numRows += 1

    return policy.reshape(numRows, numColumns)

def defineWorld(enviorment_file, ntr):
    global enviormentSize
    enviorment_file = open(enviorment_file)
    enviorment = np.array([])

    numColumns = len(enviorment_file.readline().strip().split(','))
    enviorment_file.seek(0)

    numRows = 0
    startLocation = (-1, -1)
    for rowIndex, row in enumerate(enviorment_file.readlines()):
        row = row.strip().split(',')
        temp = np.array([])

        for columnIndex, state in enumerate(row):
            state = state.strip()
            if(state == 'X'):
                temp = np.append(temp, BlockedState())
            elif(state == '.'):
                temp = np.append(temp, NonTerimnalState([rowIndex, columnIndex], ntr))
            elif(state == "I"):
                startLocation = (rowIndex, columnIndex)
                temp = np.append(temp, NonTerimnalState([rowIndex, columnIndex], ntr))
            else:
                temp = np.append(temp, TerimnalState(float(state)))

        enviorment = np.append(enviorment, temp)
        numRows += 1

    enviormentSize = [numRows, numColumns]
    return (enviorment.reshape(numRows, numColumns), startLocation)

def n(x):
    return 20 / (19 + x)

def TDL( s, r, a, s_prime, r_prime, policy, gamma, R, U, Ns):
    if(s_prime not in Ns):
        R[s_prime] = r_prime
        U[s_prime] = r_prime

    if(s != None):
        if(s in Ns):
            Ns[s] += 1
        else:
            Ns[s] = 1

        c = n(Ns[s])
        U[s] = ((1 - c) * U[s]) + (c * (R[s] + gamma * U[s_prime]))

def AgentModel_Q_Learning_Passive(enviornment_file, policy_file, ntr, gamma, number_of_moves):
    enviorment, startLocation = defineWorld(enviornment_file, ntr)
    policy = definePolicy(policy_file)

    #For debugging
    # for rowIndex, rows in enumerate(enviorment):
    #     for columnIndex, column in enumerate(rows):
    #         if(startLocation == (rowIndex, columnIndex)):
    #             print("I", end=" ")
    #         else:
    #             print(column, end=" ")
    #     print()

    # for rowIndex, rows in enumerate(enviorment):
    #     for columnIndex, column in enumerate(rows):
    #             print(column.stateReward, end=" ")
    #     print()

    # print()
    # for rows in policy:
    #     for column in rows:
    #         print(column, end=" ")
    #     print()

    # print()
    
    #Intit
    R = dict()
    U = dict()
    Ns = dict()

    k_iterations = 0
    while (k_iterations < number_of_moves):
        s = None
        r = None
        s_prime = enviorment[startLocation[0],startLocation[1]]
        a = policy[s_prime.row, s_prime.column]

        while(k_iterations < number_of_moves):
            r_prime = s_prime.senseReward()

            #print("Iteration %d" % (k_iterations + 1))
            TDL(s, r, a, s_prime, r_prime, policy, gamma, R, U, Ns)

            if(s_prime.type == "terminal"): 
                break

            a = policy[s_prime.row, s_prime.column]
            s, r = s_prime, r_prime
            s_prime = s_prime.executeAction(a, enviorment)
            k_iterations += 1

    print("utilities:")
    for row in enviorment:
        for column in row:
            if(column.type == "blocked" or column not in U):
                print("%6.3f" % 0.0, end="")
            else:
                column = U[column] 
                print("%6.3f" % column, end="")
        print()
