
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
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

## KNN ############################################################################
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("KNN [K=1]---> outputs")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Validation Curve
K_range = np.arange(1, 30)
train_scores, test_scores = validation_curve(knn, X_train, y_train, param_name="n_neighbors", param_range=K_range, cv=5)

plt.figure()
plt.plot(K_range, np.mean(train_scores, axis=1), label='Training')
plt.plot(K_range, np.mean(test_scores, axis=1), label='Testing')
plt.title('Validation for kNN')
plt.xlabel('K')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('KNN_validation_curve.jpg')


# WITH K=25
knn_best = KNeighborsClassifier(n_neighbors=25)
knn_best.fit(X_train,y_train)
predictions_best = knn_best.predict(X_test)
print("KNN [K=25]---> outputs")
print(classification_report(y_test, predictions_best))
print(confusion_matrix(y_test, predictions_best))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(knn_best, X_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Testing')
plt.title('Learning curve for kNN')
plt.xlabel('Training samples')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('KNN_learning_curve.jpg')