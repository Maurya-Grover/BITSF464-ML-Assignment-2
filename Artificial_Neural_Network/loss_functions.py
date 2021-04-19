import numpy as np


def mse(y, yhat):
    return np.mean(np.power(y - yhat, 2))


def mse_prime(y, yhat):
    return 2 * (y - yhat) / y.size