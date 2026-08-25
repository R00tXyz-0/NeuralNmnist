import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)


# Load trained weights
data = np.load("mnist_model.npz")

w1 = data["w1"]
b1 = data["b1"]
w2 = data["w2"]
b2 = data["b2"]


def predict(X):
    # Layer 1
    Z1 = w1.dot(X) + b1
    A1 = relu(Z1)

    # Output layer
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)

    prediction = np.argmax(A2, axis=0)

    return prediction, A2