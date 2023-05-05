import numpy as np

class FNNLayer:

    def __init__(self, input, output, learningRate=.01):
        self.weights = np.random.uniform(-.2,.2, [input, output])
        self.biases = np.random.uniform(-.2,.2, [1,output])
        self.input = None
        self.output = None
        self.learningRate = learningRate

    def feed_forward(self, input):
        self.input = input

        output = np.dot(input, self.weights)
        output = np.add(output, self.biases)
        output = self.activate(output)

        self.output = output
        return output

    def back_propagate(self, error):
        gradient = error * self.sigmoid_der(self.output)

        print(self.input.T)
        self.weights += self.learningRate * np.dot(self.input.T, gradient)
        self.biases += self.learningRate * np.sum(gradient, axis=0, keepdims=True)

        return np.dot(error, self.weights.T)
    
    def activate(self, x):
        return self.sigmoid(x)

    def sigmoid(self, x) -> float:
        return (1.0 / (1 + np.exp(-x)))
    
    def sigmoid_der(self, x) -> float:
        return x * (1 - x)
