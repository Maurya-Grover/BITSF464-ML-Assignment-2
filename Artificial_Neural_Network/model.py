import numpy as np
import pandas as pd
from fc_layer import FCLayer
from activation_layer import ACLayer
from activation_functions import *
from loss_functions import *
from Network import Network

df = pd.read_csv("./dataset_NN.csv")


def train_test_split(dataframe, split=0.70, randomState=1):
    train_size = int(split * len(dataframe))
    test_size = len(dataframe) - train_size
    dataframe = dataframe.sample(frac=1, random_state=randomState)
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

x_train, y_train, x_test, y_test = train_test_split(df)
net = Network()
net.add(FCLayer(x_train.shape[1], 20))
net.add(ACLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(20, 15))
net.add(ACLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(15, 10))
net.add(ACLayer(sigmoid, sigmoid_prime))

net.set_loss(mse, mse_prime)
net.fit(x_train, y_train, 10, 0.5)

out = np.array(net.predict(x_train))
y_pred = np.argmax(out, axis = 1)

print(out)
