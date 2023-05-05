import numpy as np
from FNNLayer import FNNLayer

class NN:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def feed_forward(self, input):
        for layer in self.layers:
            input = layer.feed_forward(input)
        return input

    def back_propagate(self, error):
        for layer in self.layers:
            error = layer.back_propagate(error)
    
    def mean_squared_error(self, actual, expected) -> float:
        return np.square(np.subtract(expected, actual)).mean()



nn = NN()
nn.add(FNNLayer(5, 3))
nn.add(FNNLayer(3, 2))
for i in range(100):
    input = np.array([1,1,1,1,1])
    input = np.reshape(input, [1,5])
    output = nn.feed_forward(input)
    expected = [1,0]
    print(nn.mean_squared_error(output, expected))
    nn.back_propagate(expected - output)