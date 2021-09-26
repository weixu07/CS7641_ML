import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.model_selection import GridSearchCV



## data input
loan = pd.read_csv('data/loan_data.csv')
#loan.info()
cat_feature = ['purpose']
final_data = pd.get_dummies(loan, columns = cat_feature, drop_first = True)
final_data.info()
# Train Test split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
# scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Decision Trees ##############################################################
dtree = DecisionTreeClassifier(random_state=100)
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print("Decision Tress ---> outputs")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Validation Curve
param_range = np.arange(1,21)

train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=100), X_train, y_train,
                                             param_name="max_depth", param_range=param_range, cv=5)

plt.figure()
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Testing')
plt.title('Validation for decision tree')
plt.xlabel('max_depth')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('dt_validation_curve.jpg')

# Hyperparameter Tuning
param_tuning = {'max_depth' : param_range, 'min_samples_split' : range(2, 10)}
dtree_grid = GridSearchCV(dtree, param_grid=param_tuning, cv=5)
dtree_grid.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(dtree_grid.best_params_)
predictions_best = dtree_grid.predict(X_test)

print("Dtree [max_depth = 1, min_samples_split = 2]---> outputs")
print(classification_report(y_test, predictions_best))
print(confusion_matrix(y_test, predictions_best))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(dtree_grid, X_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Testing')
plt.title('Learning curve for decision tree')
plt.xlabel('Fraction of training examples')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('dt_learning_curve.jpg')

## Plot tree #####################################################################
'''
feature_names = list(X.columns.values)
target_names = list(final_data.columns[12])
fig = plt.figure(figsize=(250,200))
_= tree.plot_tree(dtree_grid,
                  feature_names=feature_names,
                  class_names=target_names,
                  filled=True)
fig.savefig("decision_tree_plot.jpg")
'''
