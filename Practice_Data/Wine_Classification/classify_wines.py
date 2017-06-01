import display_tree as dt
import pickle
import pandas as pd
from termcolor import colored
import time
import numpy as np
from sklearn import preprocessing, tree, svm, cross_validation, linear_model, neighbors

# Import data
df = pd.read_csv('Data/wine.data')

y = np.array(df['Type'])
X = np.array(df.drop(['Type'], 1))
preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .2)

# Define the classifiers that you want to be tested
test = [[tree.DecisionTreeClassifier(), 'TREE'],
        [neighbors.KNeighborsClassifier(), 'KNN'],
        [linear_model.LinearRegression(n_jobs=-1), 'LINEAR'],
        [svm.SVC(), 'SVM']]

# Here we will set the start time for how long it takes to
# train the model

_time = time.time()

def train_and_save(clf, name):

    _time = time.time()
    print(colored('\nTraining {}...'.format(name),'green'))
    clf.fit(X_train, y_train)
    print('Elapsed: {} seconds'.format(time.time() - _time))

    with open('Saved_CLF/CLF_{}_WINE_DATA.clf'.format(name), 'wb') as f:
        f = pickle.dump(clf, f)

def accuracy(clf, x, y, name):
    print('Accuracy is: {} for {}'.format(clf.score(x,y), name))

def run(tests, x, y):
    for i in test:
        train_and_save(i[0], i[1])
        accuracy(i[0], x, y, i[1])

run(test, X_test, y_test)

dt.save_to_pdf(test[0][0], ['Alcohol','Malic','Ash','Alcalinity','Magnesium',
                            'Phenols','Flavanoids','Nonflavanoid','Proanthoc',
                            'Colorintensity','Hue','OD','Proline'],['1','2','3'], 'wine_tree')
