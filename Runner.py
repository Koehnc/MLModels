import tensorflow
import keras
import numpy as np
import matplotlib as plot

from FNN import FNN
from NE import GA

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

# Models
""" 
Trained with: 
[784,10] -> 89.53
[784,128,10] -> 91.47
[784,256,64,10] -> 92.37
"""
fnn = FNN([784,10])
""" 
Trained with: 
200 popSize, 500 gens -> 39.04
200 ps, 1000 gens -> 49.12
200 ps, 2000 gens -> 66.04
"""
ne = GA(200, [784,10])

for i in range(X_train.shape[0]):
    fnn_out = fnn.feed_forward(X_train[i].reshape(1,-1))
    fnn_cost = fnn.back_propagate(y_train[i])

    # if (i % 25 == 0):
    #     ne_cost = ne.run_gen(X_train[i].reshape(1,-1), y_train[i])

    if (i % 5000 == 0):
        print("Generation ", i, ": ")
        print("\tfnn cost: ", fnn_cost)
        # print("\tne cost: ", ne_cost)


fnn_count_right = 0
ne_count_right = 0
for i in range(X_test.shape[0]):
    fnn_out = fnn.feed_forward(X_test[i].reshape(1,-1))
    ne_out = ne.get_best().feed_forward(X_test[i].reshape(1,-1))
    if (np.argmax(fnn_out) == y_test[i]):
        fnn_count_right += 1 
        # print(out)
    if (np.argmax(ne_out) == y_test[i]):
        ne_count_right += 1 

print("FNN Accuracy: ", fnn_count_right / X_test.shape[0] * 100, "%")
print("NE Accuracy: ", ne_count_right / X_test.shape[0] * 100, "%")
