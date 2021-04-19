from gd import *
from sgd import *
from helper import *
import time
import sys
import matplotlib.pyplot as plt

start = time.time()
tot_accuracy = 0
tot_prec = 0
tot_recall = 0
tot_loss = 0
r = np.random.RandomState(1024)
algo = sys.argv[1]
print(algo)
for i in range(10):
    x_train, y_train, x_test, y_test = train_test_split(df, split = 0.7, randomState = r)
    w, b = gradient_descent(x_train, y_train, 5, 1000) if (algo == 'GD') else stochastic_gradient_descent(x_train, y_train, 5, 200)# if (algo == 'sgd') else print('error'), exit(0))
    y_pred = predict(x_train, y_train, w, b)
    accuracy = calcAccuracy(y_pred, y_train)
    print(f"Training Accuracy (Random Split #{i+1}): ", accuracy)
    y_pred = predict(x_test, y_test, w, b)
    accuracy = calcAccuracy(y_pred, y_test)
    tn =0
    tp = 0
    fn = 0
    fp = 0
    for y1,y2 in zip(y_test, y_pred):
        if y1 == 1:
            if y2 == 1:
                tp+=1
            else:
                fn+=1
        else:
            if y2==0:
                tn+=1
            else:
                fp+=1
    tot_prec+=(tn/(tp+fp))
    tot_recall += tp/ (tp + fn)
    tot_accuracy+=accuracy
    tot_loss+=cost(x_test, y_test, w, b)
    print(f"Testing Accuracy (Random Split #{i+1}): ", accuracy)
# plt.show()
end = time.time()
tot_accuracy/=10
tot_prec/=10
tot_recall/=10
tot_loss/=10
F_score=2*tot_prec*tot_recall/(tot_prec+tot_recall)
print(f"Runtime of the program is {end - start}")
print("Testing Accuracy (10 random splits) : ", tot_accuracy)
print("Testing Precision (10 random splits) : ", tot_prec)
print("Testing Recall (10 random splits) : ", tot_recall)
print("Testing F-Score (10 random splits) : ", F_score)
print("Testing loss (10 random splits) : ", tot_loss)