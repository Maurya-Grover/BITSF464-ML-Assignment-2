from gd import *
from sgd import *
from helper import *
import time
import sys

start = time.time()
algo = sys.argv[1]
x_train, y_train, x_test, y_test = train_test_split(df, 0.7)
print(algo)
w, b = gradient_descent(x_train, y_train, 0.1) if (algo == 'GD') else stochastic_gradient_descent(x_train, y_train, 0.001)# if (algo == 'sgd') else print('error'), exit(0))
y_pred = predict(x_train, y_train, w, b)
accuracy = calcAccuracy(y_pred, y_train)
print("Training Accuracy : ", accuracy)
y_pred = predict(x_test, y_test, w, b)
accuracy = calcAccuracy(y_pred, y_test)
print("Testing Accuracy : ", accuracy)
end = time.time()
print(f"Runtime of the program is {end - start}")