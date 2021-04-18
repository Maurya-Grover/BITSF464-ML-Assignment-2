
from helper import *


def stochastic_gradient_descent(x_train, y_train, alpha=0.05, num_iter=400):
    w = np.zeros(x_train.shape[1])
    b = 0
    for iter in range(num_iter):
        for i in range(x_train.shape[0]):
            x_i = x_train[i]
            y_prediction = w.dot(x_i.T) + b
            diff = sigmoid(y_prediction) - y_train[i]
            w = w - alpha * (x_i.T * diff)
            b = b - alpha * diff
    return w, b