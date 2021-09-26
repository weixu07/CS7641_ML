import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

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

## Nerual Network ############################################################################
c_nn = MLPClassifier(hidden_layer_sizes=(7, 2), random_state=7, max_iter=1000)
c_nn.fit(X_train, y_train)
predictions = c_nn.predict(X_test)
print("Nerual Network ---> outputs")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Validation Curve
Alpha_range = np.logspace(-3, 3, 7)
train_scores, test_scores = validation_curve(c_nn, X_train, y_train, param_name="alpha", param_range=Alpha_range, cv=5)
plt.figure()
plt.semilogx(Alpha_range, np.mean(train_scores, axis=1), label='Training')
plt.semilogx(Alpha_range, np.mean(test_scores, axis=1), label='Cross-validation')
plt.title('Validation for Nerual Network')
plt.xlabel('Alpha')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NerualNetwork_validation_curve_alpha.jpg')

lambda_range = np.logspace(-5, 0, 6)
train_scores, test_scores = validation_curve(c_nn, X_train, y_train, param_name="learning_rate_init", param_range=lambda_range,cv=5)
plt.figure()
plt.semilogx(lambda_range, np.mean(train_scores, axis=1), label='Training')
plt.semilogx(lambda_range, np.mean(test_scores, axis=1), label='Testing')
plt.title('Validation for Neural Network')
plt.xlabel('Lambda')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NerualNetwork_validation_curve_lambda.jpg')

# Hyperparameter tuning
alpha_range = np.logspace(-1, 2, 5)
learning_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : learning_range}

cnn_best = GridSearchCV(c_nn, param_grid=tuned_params, cv=5, n_jobs=4)
cnn_best.fit(X_train,y_train)
print("Best parameters set found on development set:")
print(cnn_best.best_params_)

predictions_best = cnn_best.predict(X_test)
print("CNN [best]---> outputs")
print(classification_report(y_test, predictions_best))
print(confusion_matrix(y_test, predictions_best))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(cnn_best, X_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Testing')
plt.title('Learning curve for kNN')
plt.xlabel('Fraction of training examples')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NerualNetwork_learning_curve.jpg')







