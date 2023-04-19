import numpy as np


class NeuralNetwork():
    # Static hyperparatmeters
    # Learning rate
    epsilon = .2


    # int[] structure
    def __init__(self, structure) -> None:
        self.weights = []
        self.biases = []
        self.input_layer_size = structure[0]
        self.layer_outputs = []

        for i in range(len(structure) - 1):
            self.weights.append(np.random.rand(structure[i], structure[i+1]))
            self.biases.append(np.random.rand(1, structure[i+1]))
    
    def feedForward(self, input) -> list:
        self.layer_outputs = [input]
        for i in range(len(self.weights)):
            # print("Multiplying: input-", input.shape, ", weight table-", self.weights[i].shape)
            input = np.dot(input, self.weights[i])
            input = np.add(input, self.biases[i])
            input = self.sigmoid(input)
            self.layer_outputs.append(input)
        
        return input
    
    def backPropogate(self, expected) -> float:
        output_error = expected - self.layer_outputs[-1]
        output_delta = output_error * self.sigmoid_der(self.layer_outputs[-1])

        hidden_error = np.dot(output_delta, self.weights[-1].T)
        hidden_delta = hidden_error * self.sigmoid_der(self.layer_outputs[-2])

        self.weights[-1] += NeuralNetwork.epsilon * np.dot(self.layer_outputs[-2].T, output_delta)
        self.biases[-1] += NeuralNetwork.epsilon * np.sum(output_delta, axis=0, keepdims=True)
        self.weights[-2] += NeuralNetwork.epsilon * np.dot(self.layer_outputs[-3].T, hidden_delta)
        self.biases[-2] += NeuralNetwork.epsilon * np.sum(hidden_delta, axis=0, keepdims=True)

        return self.mean_squared_error(self.layer_outputs[-1], expected)
    
    def mean_squared_error(self, actual, expected) -> float:
        return np.square(np.subtract(expected, actual)).mean()
    
    def sigmoid(self, x) -> float:
        return (1.0 / (1 + np.e**-x))
    
    def sigmoid_der(self, x) -> float:
        return x * (1 - x)

    def relu(self, x) -> float:
        return x * (x > 0)
    
    def relu_der(self, x) -> float:
        return (x > 0) * 1


def testNNFeedForward():
    nn = NeuralNetwork([784,256,64,10])
    print(nn.feedForward(np.random.rand(1,784)))
    print(nn.feedForward(np.full((1,784), 1)))
    print(nn.feedForward(np.full((1,784), 1)))

def testNNBackPropogate():
    nn = NeuralNetwork([10,5,4])
    input = (np.full((1,10), 1))
    expected = np.array([0,0,1,0])

    for i in range(10000):  
        out = nn.feedForward(input)
        loss = nn.backPropogate(expected)
        if (i % 1000) == 0:
            print(out)
            print(loss)
    

# testNNFeedForward()
testNNBackPropogate()