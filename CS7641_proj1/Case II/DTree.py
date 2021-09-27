import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
## data input

data = pd.read_csv('data/winequality-white.csv', sep=';')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = y.values
y = y.astype(int)
y[y < 6] = 0
y[y >= 6] = 1
#Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
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
param_range = np.arange(1,41)

train_scores, test_scores = validation_curve(DecisionTreeClassifier(random_state=100), X_train, y_train,
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

print("Dtree [best]---> outputs")
print(classification_report(y_test, predictions_best))
print(confusion_matrix(y_test, predictions_best))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(dtree_grid, X_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Testing')
plt.title('Learning curve for decision tree')
plt.xlabel('Training samples')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('dt_learning_curve.jpg')
