import time

start = time.time()
from helper import *


def descent(W, b, x_train, y_train, alpha):
    n = x_train.shape[0]
    prediction = sigmoid(np.dot(W, x_train.T) + b)
    dW = np.dot(x_train.T, (prediction - y_train.T).T) / n
    db = np.sum(prediction - y_train.T) / n
    return W - alpha * dW, b - alpha * db


def gradient_descent(x_train, y_train, alpha=0.1, num_iter=2000):
    W = np.zeros(x_train.shape[1])
    b = 0
    for iter in range(num_iter):
        W, b = descent(W, b, x_train, y_train, alpha)
    return W, b