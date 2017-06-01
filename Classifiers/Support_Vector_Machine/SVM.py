# This is something that became popular in the 80s as
# it was better than almost anything else at recognizing
# stuff like hand written digits and whatnot
# one of the most popular classifiers

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, neighbors, tree, svm

df = pd.read_csv('Data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

y = np.array(df['class'])
X = np.array(df.drop(['class'], 1))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.2)
    
clf = svm.SVC()
clf.fit(X_train, y_train)
    
accuracy = clf.score(X_test, y_test)
# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 1, 2, 3, 2, 1]])
# example_measures = example_measures.reshape(len(example_measures), -1)

# prediction = clf.predict(example_measures)
#    print('The prediction for {} is: {}'.format(example_measures, prediction))
print(accuracy)

