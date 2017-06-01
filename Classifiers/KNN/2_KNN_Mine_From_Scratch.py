import numpy as np
import warnings
import pandas as pd
import random
from collections import Counter

# Format that we are using for KNN
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

def KNN(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is to a value less than or equal to total voting groups!')

    distances = []

    for group in data:
        for features in data[group]:
            euc_dist = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euc_dist, group])

    # Get the three top votes
    votes = [i[1] for i in sorted(distances)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

# Read in our data_set
df = pd.read_csv('Data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# This is to get rid of random str() in dataset idk
full_data = df.astype(float).values.tolist()

# This is like cross_validation.x_y_split
random.shuffle(full_data)
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# Populate the dictionaries
for i in train_data:
    train_set[i[-1]].append(i[:-1]) # i[-1] is the class, remember?
for i in test_data:
    test_set[i[-1]].append(i[:-1])  # This is class, last

# Get our accuracy
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = KNN(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total)