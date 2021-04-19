from gd import *
from sgd import *
from helper import *
import time
import sys

start = time.time()
tot_accuracy = 0
r = np.random.RandomState(1)
for i in range(10):
    algo = sys.argv[1]
    x_train, y_train, x_test, y_test = train_test_split(df, split = 0.7, randomState = r)
    print(algo)
    w, b = gradient_descent(x_train, y_train, 0.1) if (algo == 'GD') else stochastic_gradient_descent(x_train, y_train, 0.001)# if (algo == 'sgd') else print('error'), exit(0))
    y_pred = predict(x_train, y_train, w, b)
    accuracy = calcAccuracy(y_pred, y_train)
    # print("Training Accuracy : ", accuracy)
    y_pred = predict(x_test, y_test, w, b)
    accuracy = calcAccuracy(y_pred, y_test)
    tot_accuracy+=accuracy
    print("Testing Accuracy : ", accuracy)

end = time.time()
print(f"Runtime of the program is {end - start}")
print("Testing Accuracy (10 random splits) : ", tot_accuracy/10)