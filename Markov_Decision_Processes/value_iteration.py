#Preston Loera, 1001889535

from abc import ABC, abstractmethod
import numpy as np

class State(ABC):
    @abstractmethod
    def utility(self, enviorment):
        pass

    @abstractmethod
    def __str__(self):
        pass

class TerimnalState(State):
    def __init__(self, reward):
        self.state_utility = reward
        self.optimal_action = "o"
    
    def utility(self , utilities, bounds, enviorment):
        return self.state_utility
    
    def __str__(self):
        return f"{self.state_utility}"

class BlockedState(State):
    def __init__(self):
        self.state_utility = 0.0
        self.optimal_action = "x"
    
    def utility(self , utilities, bounds, enviorment):
        return self.state_utility
    
    def __str__(self):
        return "x"
    
class NonTerimnalState(State):
    def __init__(self, non_terminal_reward, gamma, index):
        self.non_terminal_reward = non_terminal_reward
        self.gamma = gamma
        self.index = index
        self.row = self.index[0]
        self.column = self.index[1]
        self.optimal_action = "None"

    def up(self, utilities, bounds, enviorment):
        weightedSum = 0.0

        #Move up
        if(self.row - 1 < 0 or (enviorment[self.row - 1, self.column].optimal_action == "x")):
            weightedSum += .8 * utilities[self.row, self.column]
        else:
            weightedSum += .8 * utilities[self.row - 1, self.column]

        #Move right
        if(self.column + 1 == bounds[1] or (enviorment[self.row, self.column + 1].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row, self.column + 1]

        #Move left
        if(self.column - 1 < 0 or (enviorment[self.row, self.column - 1].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row, self.column - 1]

        return [weightedSum, "^"]

    def right(self, utilities, bounds, enviorment):
        weightedSum = 0.0

        #Move up
        if(self.row - 1 < 0 or (enviorment[self.row - 1, self.column].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row - 1, self.column]

        #Move down
        if(self.row + 1 == bounds[0] or (enviorment[self.row + 1, self.column].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row + 1, self.column]

        #Move right
        if(self.column + 1 == bounds[1] or (enviorment[self.row, self.column + 1].optimal_action == "x")):
            weightedSum += .8 * utilities[self.row, self.column]
        else:
            weightedSum += .8 * utilities[self.row, self.column + 1]

        return [weightedSum, ">"]

    def down(self, utilities, bounds, enviorment):
        weightedSum = 0.0

        #Move right
        if(self.column + 1 == bounds[1] or (enviorment[self.row, self.column + 1].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row, self.column + 1]

        #Move down
        if(self.row + 1 == bounds[0] or (enviorment[self.row + 1, self.column].optimal_action == "x")):
            weightedSum += .8 * utilities[self.row, self.column]
        else:
            weightedSum += .8 * utilities[self.row + 1, self.column]

        #Move left
        if(self.column - 1 < 0 or (enviorment[self.row, self.column - 1].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row, self.column - 1]

        return [weightedSum, "v"]
        
    def left(self, utilities, bounds, enviorment):
        weightedSum = 0.0

        #Move up
        if(self.row - 1 < 0 or (enviorment[self.row - 1, self.column].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row - 1, self.column]

        #Move down
        if(self.row + 1 == bounds[0] or (enviorment[self.row + 1, self.column].optimal_action == "x")):
            weightedSum += .1 * utilities[self.row, self.column]
        else:
            weightedSum += .1 * utilities[self.row + 1, self.column]

        #Move left
        if(self.column - 1 < 0 or (enviorment[self.row, self.column - 1].optimal_action == "x")):
            weightedSum += .8 * utilities[self.row, self.column]
        else:
            weightedSum += .8 * utilities[self.row, self.column - 1]

        return [weightedSum, "<"]

    def bellman(self, utilities, bounds, enviorment):
        util, self.optimal_action = max([self.up(utilities, bounds, enviorment), self.right(utilities, bounds, enviorment), 
                                         self.down(utilities, bounds, enviorment), self.left(utilities, bounds, enviorment)])

        return self.non_terminal_reward + (self.gamma * (util))

    def utility(self , utilities, bounds, enviorment):
        return self.bellman(utilities, bounds, enviorment)

    def rewardOfState(self):
        return self.non_terminal_reward
    
    def __str__(self):
        return "."

def defineWorld(enviorment_file, ntr, gamma):
    enviorment_file = open(enviorment_file)
    enviorment = np.array([])

    numColumns = len(enviorment_file.readline().strip().split(','))
    enviorment_file.seek(0)

    numRows = 0
    for rowIndex, row in enumerate(enviorment_file.readlines()):
        row = row.strip().split(',')
        temp = np.array([])

        for columnIndex, state in enumerate(row):
            if(state == 'X'):
                temp = np.append(temp, BlockedState())
            elif(state == '.'):
                temp = np.append(temp, NonTerimnalState(ntr, gamma, [rowIndex, columnIndex]))
            else:
                temp = np.append(temp, TerimnalState(float(state)))

        enviorment = np.append(enviorment, temp)
        numRows += 1

    return enviorment.reshape(numRows, numColumns)

def value_iteration(data_file, ntr, gamma, K):
    enviorment = defineWorld(data_file, ntr, gamma)
    
    dimensions = enviorment.shape
    uPrime = np.zeros((dimensions[0], dimensions[1]), dtype=float)

    for k in range(K):
        u = uPrime.copy()

        for rowIndex, row in enumerate(enviorment):
            for columnIndex, column in enumerate(row):
                uPrime[rowIndex, columnIndex] = column.utility(u, dimensions, enviorment)

    print("utilities:")
    for row in uPrime:
        for column in row:
            print("%6.3f" % column, end="")
        print()

    print()
    print("policy:")
    for row in enviorment:
        for column in row:
            print("%6s" % column.optimal_action, end="")
        print()