import numpy as np
import random

class NE:

    # int[] structure
    def __init__(self, structure) -> None:
        self.error = 0
        self.weights = []
        self.biases = []
        self.input_layer_size = structure[0]
        self.layer_outputs = []

        for i in range(len(structure) - 1):
            self.weights.append(np.random.uniform(-.2, .2, (structure[i], structure[i+1])))
            self.biases.append(np.zeros((1, structure[i+1])))


    def feed_forward(self, input, print_bool = False) -> list:
        self.layer_outputs = [input]
        if (print_bool): print("\n")
        for i in range(len(self.weights)):
            # print("Multiplying: input-", input.shape, ", weight table-", self.weights[i].shape)
            input = np.dot(input, self.weights[i])
            input = np.add(input, self.biases[i])
            input = self.sigmoid(input)
            if (print_bool): print(input)
            self.layer_outputs.append(input)

        self.output = input
        return input
    
    def sigmoid(self, x) -> float:
        return (1.0 / (1 + np.exp(-x)))
    
    def mean_squared_error(self, expected) -> float:
        self.error = np.square(np.subtract(expected, self.output)).mean()
        return np.square(np.subtract(expected, self.output)).mean()

    def __lt__(self, other):
        return self.error < other.error
    



class GA:
    mutRate = 0.05
    mutScale = .02


    def __init__(self, popSize, structure):
        self.structure = structure
        self.popSize = popSize
        self.pop = []
        for i in range(popSize):
            self.pop.append(NE(structure))

    def run_gen(self, input, expected):
        for nn in self.pop:
            nn.feed_forward(input)
            nn.mean_squared_error(expected)

        # for nn in self.pop:
        #     print(nn.error)
        # print()
        # print(self.average_error())

        self.remove_worst(self.popSize // 2)
        
        while len(self.pop) < self.popSize:
            selected = self.select(2, 10)
            kids = self.crossover(selected[0], selected[1])

            for i in range(len(kids)):
                self.mutate(kids[i])
                self.pop.append(kids[i])

        return self.average_error()


    def select(self, numSel, rounds):
        chosen = []
        for i in range(numSel):
            chosen.append(self.tournament_select(rounds))
            self.pop.remove(chosen[-1])

        for i in range(len(chosen)):
            self.pop.append(chosen[i])
        return chosen

    def tournament_select(self, rounds):
        chosen = random.choice(self.pop)
        for i in range(rounds):
            chance = random.choice(self.pop)

            if chance.error < chosen.error:
                chosen = chance

        return chosen

    def crossover(self, parent1, parent2):
        child1 = NE(self.structure)
        child2 = NE(self.structure)
        # Copying issues in python is a bitch. Java is better
        child1.weights = [np.copy(weights) for weights in parent1.weights]
        child2.weights = [np.copy(weights) for weights in parent2.weights]

        for i in range(len(parent1.weights)):
            choices = np.random.choice([False, True], size=parent1.weights[i].shape)
            child2.weights[i][choices] = parent1.weights[i][choices]
            child1.weights[i][choices] = parent2.weights[i][choices]

        return [child1, child2]

    def mutate(self, nn):
        for i in range(len(nn.weights)):
            for j in range(len(nn.weights[i])):
                for k in range(len(nn.weights[i][j])):
                    if random.uniform(0, 1) < GA.mutRate:
                        nn.weights[i][j][k] += random.gauss(0, GA.mutScale)


    def remove_worst(self, amount):
        self.pop.sort()
        self.pop = self.pop[:amount]

    def average_error(self):
        sum = 0
        for i in range(len(self.pop)):
            sum += self.pop[i].error
        return sum / len(self.pop)

    def get_best(self):
        self.pop.sort()
        return self.pop[0]

genetic = GA(5, [5,2])
for i in range(100):
    genetic.run_gen([1, 0, 0, 0, 0], [0, 1])

