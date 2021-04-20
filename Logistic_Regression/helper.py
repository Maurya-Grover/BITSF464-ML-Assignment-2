import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./dataset_LR.csv")


def train_test_split(dataframe, split=0.70, randomState = 1):
    train_size = int(split * len(dataframe))
    test_size = len(dataframe) - train_size
    dataframe = dataframe.sample(frac=1, random_state = randomState)
    train = dataframe[:train_size].to_numpy()
    test = dataframe[-test_size:].to_numpy()
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]
    return x_train, y_train, x_test, y_test


def plot(x_train, y_train):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x_train[y_train == 1, 1],
        x_train[y_train == 1, 2],
        x_train[y_train == 1, 3],
        color="blue",
    )
    ax.scatter(
        x_train[y_train == 0, 1],
        x_train[y_train == 0, 2],
        x_train[y_train == 0, 3],
        color="red",
    )
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(x_test, y_test, w, b):
    predicted_value = x_test.dot(w) + b
    yhat = sigmoid(predicted_value)
    return yhat >= 0.5

def cost(x, y, w, b):
    predicted_value = x.dot(w) + b
    yhat = sigmoid(predicted_value)
    return -np.sum(np.dot(y, np.log(yhat)) + np.dot(1-y, np.log(1-yhat)))/x.shape[0]


def calcAccuracy(y_pred, y_test):
    check_pred = np.array([y_pred == y_test]).T
    accuracy = (np.sum(check_pred) * 100) / check_pred.shape[0]
    return accuracy

