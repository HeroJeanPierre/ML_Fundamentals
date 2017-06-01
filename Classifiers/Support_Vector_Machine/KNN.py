import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import KNNeighbors
from sklearn import preprocessing, cross_validation, neighbors, tree

# Set up data
df = pd.read_csv('breast-cancer-wisconsin.data')

#print(df.head())               
df.replace('?', -99999, inplace=True)
# We need to drop the features that do now contribute
# to the result df.dropna(inplace=True)
# df.dropna(inplace=True)
df.drop(['id'], 1, inplace=True)

y = np.array(df['class'])
X = np.array(df.drop(['class'], 1))

# print(X)
# print(y)
accuracies = []
for i in range(5):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.2)
    
    # Test the data
    clf = neighbors.KNeighborsClassifier(n_jobs=1)
    clf.fit(X_train, y_train)
    
    # with open('KNNWis consinBreastCancer.clf', 'wb') as f:
    #     pickle.dump(clf, f)
    

    accuracy = clf.score(X_test, y_test)
    # print(accuracy)
    
    # # plt.scatter(df['class'], df['mitoses'])
    # # plt.show()
    
    # example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 1, 2, 3, 2, 1]])
    # example_measures = example_measures.reshape(len(example_measures), -1)
    
    # prediction = clf.predict(example_measures)
    # print('The prediction for {} is: {}'.format(example_measures, prediction))

    accuracies.append(accuracy)
    print(accuracy)
print(sum(accuracies) /len(accuracies))
