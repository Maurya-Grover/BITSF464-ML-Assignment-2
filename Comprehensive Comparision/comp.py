import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

df = pd.read_csv("./dataset_comb.csv")
df.Class[df.Class == 'jasmine'] = 0
df.Class[df.Class == 'Gonen'] = 1

X = df[df.columns[1:-1]]
y = df[df.columns[-1]]
y = y.astype(int)

LDAscores = []
PCPscores = []
NBscores = []
LRscores = []
ANNscores = []
SVMscores = []

for i in range(7):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    LDAmodel = LinearDiscriminantAnalysis()
    LDAmodel.fit(X_train, y_train)
    LDAscores.append(LDAmodel.score(X_test, y_test, sample_weight=None))
    
    PCPmodel = Perceptron()
    PCPmodel.fit(X_train, y_train)
    y_pred = PCPmodel.predict(X_test)
    PCPscores.append(accuracy_score(y_test, y_pred))
    
    NBmodel = GaussianNB()
    NBmodel.fit(X_train, y_train)
    NBscores.append(NBmodel.score(X_test, y_test))
    
    LRmodel = LogisticRegression()
    LRmodel.fit(X_train, y_train)
    LRscores.append(LRmodel.score(X_test, y_test))
    
    ANNmodel = MLPClassifier(random_state=1, max_iter=300)
    ANNmodel.fit(X_train, y_train)
    ANNscores.append(ANNmodel.score(X_test, y_test))
    
    SVMmodel = SVC(kernel = 'linear')
    SVMmodel.fit(X_train, y_train)
    y_pred = SVMmodel.predict(X_test)
    SVMscores.append(accuracy_score(y_test, y_pred))

fig = plt.figure(figsize =(10, 7))
ax = fig.add_axes([1, 0, 1, 1])
data = [np.array(LDAscores), np.array(PCPscores), np.array(NBscores), np.array(LRscores), np.array(ANNscores), np.array(SVMscores)]

bp = ax.boxplot(data)
ax.set_xticklabels(['Fisher_LDA', 'Perceptron', 'Naive_Bayes', 'Logistic_Regression', 'ANN', 'SVM'])
plt.title("Variation of Accuracy over each fold")
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.show(bp)
