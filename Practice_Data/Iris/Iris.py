import time
import numpy as np
import pickle
from termcolor import colored
import pandas as pd
from sklearn import tree, svm, neighbors, linear_model, cross_validation, preprocessing

# Import the data
df = pd.read_csv('Data/iris.data')

# Set up the X and y data
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .2)

tests = [[tree.DecisionTreeClassifier(), 'TREE'],
         [neighbors.KNeighborsClassifier(), 'KNN'],
#        [linear_model.LinearRegression(n_jobs=-1), 'LINEAR'], 
         [svm.SVC(), 'SVM']]


_time = time.time()

def train_and_save(clf, name):

    _time = time.time()
    print(colored('\nTraining {}...'.format(name),'green'))
    clf.fit(X_train, y_train)
    print('Elapsed: {} seconds'.format(time.time() - _time))

    with open('Saved_CLF/CLF_{}_IRIS_DATA.clf'.format(name), 'wb') as f:
        f = pickle.dump(clf, f)

def accuracy(clf, x, y, name):
    print('Accuracy is: {} for {}'.format(clf.score(x,y), name))


def run(test , x, y):
    for i in test:
        train_and_save(i[0], i[1])
        accuracy(i[0], x, y, i[1])


run(tests, X_test, y_test)
