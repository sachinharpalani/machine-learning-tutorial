import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from adspy_shared_utilities import plot_fruit_knn

fruits = pd.read_table('fruits_data_with_colors.txt')

# Creating train-test split
X = fruits[['mass','width','height','color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

# Visualizing Data
# plotting a scatter matrix
# cmap = cm.get_cmap('gnuplot')
# scatter = pd.plotting.scatter_matrix(x_train,c=y_train,marker='o',s=40,hist_kwds={'bins':50},figsize=(12,12),cmap=cmap)
#plt.show()

# plotting a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(X_train['width'],X_train['height'],X_train['color_score'],c=y_train,marker='o',s=100)
# ax.set_xlabel('width')
# ax.set_ylabel('height')
# ax.set_zlabel('color_score')
# plt.show()

# Creating Classifier object
knn = KNeighborsClassifier() #by default knn has n_neighbors=5

# Train the classifier using the training set
knn.fit(X_train,y_train)

# Estimate the accuracy of the classifier on future data, using the test data
accuracy = knn.score(X_test,y_test)

print(accuracy)

# Use the trained k-NN classifier model to classify new, previously unseen objects

fruit_prediction = knn.predict([[20,4.3,5.5,0.78]])
print(lookup_fruit_name[fruit_prediction[0]])

# Plot the decision boundaries of the k-NN classifier
#plot_fruit_knn(X_train, y_train, 5, 'uniform')
#plot_fruit_knn(X_train, y_train, 5, 'distance')

# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
# k_range = range(1,20)
# scores = []

# for k in k_range:
#     knn = kneighborsclassifier(n_neighbors = k)
#     knn.fit(x_train, y_train)
#     scores.append(knn.score(x_test, y_test))

# plt.figure()
# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.scatter(k_range, scores)
# plt.xticks([0,5,10,15,20]);
# plt.show()

# How sensitive is k-NN classification accuracy to the train/test split proportion?
# t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# knn = KNeighborsClassifier(n_neighbors = 5)

# plt.figure()

# for s in t:

#     scores = []
#     for i in range(1,1000):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
#         knn.fit(X_train, y_train)
#         scores.append(knn.score(X_test, y_test))
#     plt.plot(s, np.mean(scores), 'bo')

# plt.xlabel('Training set proportion (%)')
# plt.ylabel('accuracy');
# plt.show()
