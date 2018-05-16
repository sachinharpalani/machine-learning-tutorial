import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

dataset = load_digits()
X,y = dataset.data, dataset.target

# for class_name, class_count in zip(dataset.target_names,np.bincount(dataset.target)):
#     print(class_name, class_count)

y_binary_inbalanced = y.copy()
y_binary_inbalanced[y_binary_inbalanced != 1] = 0

print('Original labels:',y[1:30])
print('New binary labels:',y_binary_inbalanced[1:30])

#print(np.bincount(y))
#print(np.bincount(y_binary_inbalanced))

X_train, X_test, y_train, y_test = train_test_split(X,y_binary_inbalanced,random_state=0)

svm = SVC(kernel='rbf',C=1).fit(X_train, y_train)
acc = svm.score(X_test,y_test)

print('Accuracy of SVM using RBF Kernel:',acc)

dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
y_dummy_predictions = dummy_majority.predict(X_test)

#print(y_dummy_predictions)

dummy_acc = dummy_majority.score(X_test,y_test)

print('Accuracy of dummy classifier',dummy_acc)


svm = SVC(kernel='linear',C=1).fit(X_train, y_train)
acc1 = svm.score(X_test,y_test)

print('Accuracy of SVM using linear Kernel:',acc1)

#Confusion Matrix

dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
y_majority_predicted = dummy_majority.predict(X_test)

confusion = confusion_matrix(y_test,y_majority_predicted)

print('Most frequent class (dummy classifier) : \n',confusion)

dummy_classprop = DummyClassifier(strategy='stratified').fit(X_train,y_train)
y_classprop_predicted = dummy_classprop.predict(X_test)

confusion_classprop = confusion_matrix(y_test,y_classprop_predicted)

print('Random Class proportion prediction (dummy classifier) : \n',confusion_classprop)
svm = SVC(kernel='linear',C=1).fit(X_train, y_train)
svm_predict = svm.predict(X_test)
confusion_svm = confusion_matrix(y_test,svm_predict)

print('SVM Prediction \n', confusion_svm)

lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)

dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)

print('Decision tree classifier (max_depth = 2)\n', confusion)
