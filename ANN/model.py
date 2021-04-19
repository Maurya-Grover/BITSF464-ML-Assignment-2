import pandas as pd
import numpy as np

df = pd.read_csv("./dataset_NN.csv")


def train_test_split(dataframe, split=0.70, randomState=1):
    train_size = int(split * len(dataframe))
    test_size = len(dataframe) - train_size
    dataframe = dataframe.sample(frac=1, random_state=randomState)
    train = dataframe[:train_size].to_numpy()
    train = normalise(train)
    test = dataframe[-test_size:].to_numpy()
    x_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int)
    x_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int)
    return x_train, y_train, x_test, y_test


def expand_y(y):
    b = np.zeros((y.size, y.max()))
    b[np.arange(y.size), y - 1] = 1
    return b


def normalise(X):
    for i in range(3, 6):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    return X


x_train, y_train, x_test, y_test = train_test_split(df, split=0.7)
y_train_encoded = expand_y(y_train)
y_test_encoded = expand_y(y_test)
# x_train = normalise(x_train)
# x_test = normalise(x_test)


def relu(z):
    return np.maximum(0, z)


def relu_derv(z):
    return 1 * (z > 0)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_derv(s):
    return s * (1 - s)


def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res / n_samples


def error(pred, real):
    n_samples = real.shape[0]
    logp = -np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss


class MyNN:
    def __init__(self, x, y, lr, nnl1, nnl2):
        self.x = x
        self.y = y
        self.lr = lr
        self.nnl1 = nnl1
        self.nnl2 = nnl2
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, self.nnl1) * np.sqrt(2 / ip_dim)
        self.b1 = np.zeros((1, self.nnl1))
        self.w2 = np.random.randn(self.nnl1, self.nnl2) * np.sqrt(2 / nnl1)
        self.b2 = np.zeros((1, self.nnl2))
        self.w3 = np.random.randn(self.nnl2, op_dim) * np.sqrt(2 / nnl2)
        self.b3 = np.zeros((1, op_dim))

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = relu(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = relu(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self):
        loss = error(self.a3, self.y)
        print("Error :", loss)
        a3_delta = cross_entropy(self.a3, self.y)  # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * relu_derv(self.a2)  # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * relu_derv(self.a1)  # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3


model = MyNN(x_train, y_train_encoded, 0.01, 15, 10)

epochs = 1000
for epoch in range(epochs):
    model.feedforward()
    model.backprop()


def accuracy(y, yhat):
    return np.sum((y == yhat).astype(np.int)) / y.shape[0]


yhat_train = model.predict(x_train)
yhat_test = model.predict(x_test)
train_accuracy = accuracy(np.argmax(yhat_train, axis=1) + 1, y_train)
test_accuracy = accuracy(np.argmax(yhat_test, axis=1) + 1, y_test)

# yhat_train = np.argmax(model.predict(x_train), axis = 1) + 1
# yhat_test = np.argmax(model.predict(x_test), axis = 1) + 1
print("Training accuracy : ", train_accuracy)
print("Test accuracy : ", test_accuracy)


"""

def get_acc(x, y):
    acc = 0
    for xx, yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc += 1
    return acc / len(x) * 100"""
