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

    def senseReward(self):
        return self.stateReward

def defineActions(actions_file):
    actions = []
    actions_file = open(actions_file)

    for action_line in actions_file.readlines():
        action_line = action_line.strip("\n").split(", ")
        actions.append(action_line)
    
    # for action in actions:
    #     print(action)

    return actions

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

def max_Q_sa(state, Q):
    if(state.type == "terminal"): return state.stateReward
    
    return max([Q[(state, "^")], Q[(state, ">")], Q[(state, "v")], Q[(state, "<")]])

def n(x):
    return 20 / (19 + x)

def Q_Learning_Update(s, r, a, s_prime, r_prime, gamma, Q, N_sa):
    if (s_prime.type == "terminal"):
        Q[(s_prime, None)] = r_prime
    
    if (s != None):
        if((s, a) in N_sa):
            N_sa[(s, a)] += 1
        else:
            N_sa[(s, a)] = 1

        c = n(N_sa[(s, a)])

        Q[(s,a)] = ((1-c) * Q[(s,a)]) + (c * (r + gamma * max_Q_sa(s_prime, Q)))

def initQ(enviorment):
    Q = dict()
    for row in enviorment:
        for column in row:
            if(column.type == "terminal"):
                Q[(column, None)] = 0.0
            elif(column.type == "blocked"):
                continue
            else:
                Q[(column, "^")] = 0.0
                Q[(column, ">")] = 0.0
                Q[(column, "v")] = 0.0
                Q[(column, "<")] = 0.0
    return Q

def AgentModel_Q_Learning_Deterministic(enviornment_file, actions_file, ntr, gamma, number_of_moves):
    enviorment, startLocation = defineWorld(enviornment_file, ntr)
    actions = defineActions(actions_file)
    
    #Intit
    Q = initQ(enviorment)
    N_sa = dict()

    k_iterations = 0
    while (k_iterations != len(actions) and k_iterations < number_of_moves):
        a = None
        s = None
        r = None
        s_prime = enviorment[startLocation[0],startLocation[1]]

        while(k_iterations != len(actions) and k_iterations < number_of_moves):
            r_prime = s_prime.senseReward()

            Q_Learning_Update(s, r, a, s_prime, r_prime, gamma, Q, N_sa)

            if s_prime.type == "terminal":
                break

            current_row, current_col, action, next_row, next_col = actions[k_iterations]

            s = enviorment[int(current_row), int(current_col)]
            a = action
            r = r_prime                       
            s_prime = enviorment[int(next_row), int(next_col)]

            k_iterations += 1

    print("utilities:")
    for row in enviorment:
        for column in row:
            if(column.type == "blocked"):
                print("%6.3f " % 0.0, end="")
            elif(column.type == "terminal"):
                print("%6.3f " % Q[column, None], end="")
            else:
                print("%6.3f " % max([Q[(column, "^")], Q[(column, ">")], Q[(column, "v")], Q[(column, "<")]]), end="")
        print()

    # print("policy:")
    # for row in enviorment:
    #     for column in row:
    #         if(column.type == "blocked"):
    #             print("%6.3s" % "x", end="")
    #         elif(column.type == "terminal"):
    #             print("%6.3s" % "o", end="")
    #         else:
    #             temp = []

    #             if((column, "^") in Q): temp.append((Q[(column, "^")], "^"))

    #             if((column, ">") in Q): temp.append((Q[(column, ">")], ">"))

    #             if((column, "v") in Q): temp.append((Q[(column, "v")], "v"))

    #             if((column, "<") in Q): temp.append((Q[(column, "<")], "<"))

    #             utility = max(temp)
    #             print("%6.3s" % utility[1], end="")
    #     print()