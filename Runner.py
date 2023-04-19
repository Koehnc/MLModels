import tensorflow
import keras
import numpy as np
import matplotlib as plot

from NN import NeuralNetwork


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(True)
y_train = np.array([np.array([int(i==j) for j in range(10)]) for i in y_train])
print(X_train.shape)
print(y_train.shape)

nn = NeuralNetwork([784,16,16,10])
for i in range(X_train.shape[0]):
    out = nn.feed_forward(X_train[i].reshape(1,-1))
    cost = nn.back_propagate(y_train[i])

    if (i % 5000 == 0):
        print("Generation ", i, ": ", cost)
        print(out)


count_right = 0
for i in range(X_test.shape[0]):
    out = nn.feed_forward(X_test[i].reshape(1,-1))
    if (np.argmax(out) == y_test[i]):
        count_right += 1

print("Accuracy: ", count_right / X_test.shape[0] * 100, "%")
    

