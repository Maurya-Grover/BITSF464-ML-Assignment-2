import numpy as np


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return max(0, z)


def relu_prime(z):
    return int(z > 0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return (sigmoid(z) * (1 - sigmoid(z)))
