from helper import *



def sgd(x_train, y_train, alpha=0.05, num_iter=400):
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

x_train, y_train, x_test, y_test = train_test_split(df, 0.7)
w, b= sgd(x_train, y_train, 0.001)
y_pred = predict(x_train, y_train, w, b)
accuracy = calcAccuracy(y_pred, y_train)
print("Training Accuracy : ", accuracy)
y_pred = predict(x_test, y_test, w, b)
accuracy = calcAccuracy(y_pred, y_test)
print("Testing Accuracy : ", accuracy)