import random
import numpy as np
import pandas as pd
import warnings
from collections import Counter

# The data that is passed through will look like this
data = {2: [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], 4: [[2, 3, 45, 5, 2, 2], [1, 2, 3, 4, 5], [3, 4, 4, 4, 4]]}
predict = [1, 2, 3, 4, 5]


# (1) Function that returns the predicted class w/ KNN
def KNN(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K should be be greater than the number of groups...')

    # Holds distance and group
    distances = []

    for group in data:
        for features in data[group]:
            distances.append([np.linalg.norm(np.array(features) - np.array(predict)), group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result


# (2) Import the data
df = pd.read_csv('Data/breast-cancer-wisconsin.data')

# (3) Set up the data
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_dataset = df.astype(float).values.tolist()
# df = list(df.astype(float))
accuracies = []
for i in range(5):
    # cross_validation
    random.shuffle(full_dataset)
    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}

    train_data = full_dataset[:int(len(full_dataset) * test_size)]
    test_data = full_dataset[int(len(full_dataset) * test_size):]

    [train_set[i[-1]].append(i[:-1]) for i in full_dataset]
    [test_set[i[-1]].append(i[:-1]) for i in full_dataset]

    # (4) Test accuracy
    correct, total = 0, 0

    for group in test_set:
        for predict in test_set[group]:
            vote = KNN(train_set, predict, k=5)
            if(vote == group):
                correct += 1
            total += 1

    # (5) Print Results
    print('Accuracy:', correct/total)
    accuracies.append(correct/total)
print(sum(accuracies)/len(accuracies))
