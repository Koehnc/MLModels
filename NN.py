import numpy as np

class NeuralNetwork():
    # Static hyperparatmeters
    # Learning rate
    epsilon = .01


    # int[] structure
    def __init__(self, structure) -> None:
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

        return input
    
    def back_propagate(self, expected) -> float:
        hidden_error = expected - self.layer_outputs[-1]
        hidden_delta = hidden_error * self.sigmoid_der(self.layer_outputs[-1])

        self.weights[-1] += NeuralNetwork.epsilon * np.dot(self.layer_outputs[-2].T, hidden_delta)
        self.biases[-1] += NeuralNetwork.epsilon * np.sum(hidden_delta, axis=0, keepdims=True)

        for i in range(1, len(self.weights)):
            hidden_error = np.dot(hidden_error, self.weights[-i].T)
            hidden_delta = hidden_error * self.sigmoid_der(self.layer_outputs[-(i+1)])

            self.weights[-(i+1)] += NeuralNetwork.epsilon * np.dot(self.layer_outputs[-(i+2)].T, hidden_delta)
            self.biases[-(i+1)] += NeuralNetwork.epsilon * np.sum(hidden_delta, axis=0, keepdims=True)

        return self.mean_squared_error(self.layer_outputs[-1], expected)
    
    def mean_squared_error(self, actual, expected) -> float:
        return np.square(np.subtract(expected, actual)).mean()
    
    def sigmoid(self, x) -> float:
        return (1.0 / (1 + np.exp(-x)))
    
    def sigmoid_der(self, x) -> float:
        return x * (1 - x)

    def relu(self, x) -> float:
        return np.maximum(x, 0.00)
    
    def relu_der(self, x) -> float:
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def softmax_der(self, z):
        s = self.softmax(z)
        return s * (np.identity(len(s)) - s)

    def cross_entropy_loss(self, y_pred, y_true):
        # Clip the predicted values to avoid log(0) errors
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)

        # Compute the cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

        return loss