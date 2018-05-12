import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
new_features = [5,7]

#To visualize dataset and new_features
# [[plt.scatter(j[0],j[1],s=100,color=i) for j in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1],s=100,color='g')
# plt.show()

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            #This is equivalent to:
            # euclidean_distance = np.sqrt(np.sum((np.array(featues) - np.array(predict))**2))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset,new_features)
print(result)

[[plt.scatter(j[0],j[1],s=100,color=i) for j in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],s=200,color=result)
plt.show()
