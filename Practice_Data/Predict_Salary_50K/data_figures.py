from termcolor import colored
import pickle
import time
import pandas as pd
import numpy as np
from sklearn import cross_validation, neighbors, tree, svm, linear_model, preprocessing
df = pd.read_csv('Data/adult.data')
values = {}

# Create a list of features

# If the catigory is not a string there is not point in creating
# list of features for them

for catigory in df:
    if type(df[catigory][1]) is not str:
        print('Deleting {}, because: {}'.format(catigory, df[catigory][1]))
        df.drop([catigory], 1, inplace=True)

for catigory in df:
    values[catigory] = []
    for data in df[catigory]:
        if data not in values[catigory]:
            values[catigory].append(data)

df = pd.read_csv('Data/adult.data')

# Function to convert all the strings in df to integers

for key in values.keys():
    for i, items in enumerate(values[key]):
        df.replace(items, i, inplace=True)

# Set up data
y = np.array(df['earnings'])
X = np.array(df.drop(['earnings'], 1))
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size= .2)

test = [[tree.DecisionTreeClassifier(), 'TREE'],
        [neighbors.KNeighborsClassifier(), 'KNN'],
        [linear_model.LinearRegression(n_jobs=-1), 'LINEAR']]

_time = time.time()

def train_and_save(clf, name):
    
    _time = time.time()
    print(colored('\nTraining {}...'.format(name),'green'))
    clf.fit(X_train, y_train)
    print('Elapsed: {} seconds'.format(time.time() - _time))

    with open('Saved_CLF/CLF_{}_ADULT_DATA.clf'.format(name), 'wb') as f:
        f = pickle.dump(clf, f)

def accuracy(clf, x, y, name):
    print('Accuracy is: {} for {}'.format(clf.score(x,y), name))

def run(tests, x, y):
    for i in test:
        train_and_save(i[0], i[1])
        accuracy(i[0], x, y, i[1])

run(test, X_test, y_test)
