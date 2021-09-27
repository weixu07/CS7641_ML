import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

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

## Nerual Network ############################################################################
c_nn = MLPClassifier(hidden_layer_sizes=(3, 1), random_state=7, max_iter=1000)
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

# WITH alpha = 0, lambda = 10**-3
cnn_best = MLPClassifier(hidden_layer_sizes=(3, 1), random_state=7, max_iter=1000, alpha= 10**-1, learning_rate_init=10**-3)
cnn_best.fit(X_train,y_train)
predictions_best = cnn_best.predict(X_test)
print("CNN [alpha = 0, lambda = 10**-3]---> outputs")
print(classification_report(y_test, predictions_best))
print(confusion_matrix(y_test, predictions_best))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(cnn_best, X_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Testing')
plt.title('Learning curve for kNN')
plt.xlabel('Training samples')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NerualNetwork_learning_curve.jpg')



