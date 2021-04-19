from gd import *
from sgd import *
from helper import *
import time
import sys
import matplotlib.pyplot as plt

start = time.time()
tot_accuracy = 0
r = np.random.RandomState(1024)
algo = sys.argv[1]
print(algo)
for i in range(10):
    x_train, y_train, x_test, y_test = train_test_split(df, split = 0.7, randomState = r)
    w, b = gradient_descent(x_train, y_train, 5, 1000) if (algo == 'GD') else stochastic_gradient_descent(x_train, y_train, 0.001, 200)# if (algo == 'sgd') else print('error'), exit(0))
    y_pred = predict(x_train, y_train, w, b)
    accuracy = calcAccuracy(y_pred, y_train)
    print(f"Training Accuracy (Random Split #{i+1}): ", accuracy)
    y_pred = predict(x_test, y_test, w, b)
    accuracy = calcAccuracy(y_pred, y_test)
    tot_accuracy+=accuracy
    print(f"Testing Accuracy (Random Split #{i+1}): ", accuracy)
plt.show()
end = time.time()
print(f"Runtime of the program is {end - start}")
print("Testing Accuracy (10 random splits) : ", tot_accuracy/10)