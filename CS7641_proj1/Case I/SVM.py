import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


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


## Support Vector Machines #######################################################
svc_linear = SVC(kernel='linear')   # linear kernel
svc_linear.fit(X_train, y_train)
predictions_linear = svc_linear.predict(X_test)
print("Support Vector Machines[linear] ---> outputs")
print(classification_report(y_test, predictions_linear))
print(confusion_matrix(y_test, predictions_linear))

svc_sig = SVC(kernel='sigmoid')  # sigmoid kernel
svc_sig.fit(X_train, y_train)
predictions_sig = svc_sig.predict(X_test)
print("Support Vector Machines[sigmoid] ---> outputs")
print(classification_report(y_test, predictions_sig))
print(confusion_matrix(y_test, predictions_sig))

# Validation Curve_linear
C_range = np.logspace(-3, 3, 5)
train_scores, test_scores = validation_curve(svc_linear, X_train, y_train, param_name="C", param_range=C_range, cv=5)

plt.figure()
plt.semilogx(C_range, np.mean(train_scores, axis=1), label='Training')
plt.semilogx(C_range, np.mean(test_scores, axis=1), label='Testing')
plt.title('Validation curve for SVM (linear kernel)')
plt.xlabel('C')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('SVM_validation_curve_linear.jpg')

# Validation Curve_Sigmoid
gamma_range = [1, 0.1, 0.01, 0.001, 0.0001]
train_scores, test_scores = validation_curve(svc_sig, X_train, y_train, param_name="gamma", param_range=gamma_range, cv=5)

plt.figure()
plt.semilogx(gamma_range, np.mean(train_scores, axis=1), label='Training')
plt.semilogx(gamma_range, np.mean(test_scores, axis=1), label='Testing')
plt.title('Validation for SVM (Sigmoid kernel)')
plt.xlabel('gamma')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('SVM_validation_curve_sig.jpg')

# WITH gamma=10^-1
svc_sig = SVC(kernel='sigmoid', gamma=10**-2)  # sigmoid kernel
svc_sig.fit(X_train, y_train)
predictions_sig = svc_sig.predict(X_test)
print("Support Vector Machines[sigmoid] ---> outputs")
print(classification_report(y_test, predictions_sig))
print(confusion_matrix(y_test, predictions_sig))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(svc_sig, X_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Testing')
plt.title('Learning curve for SVM')
plt.xlabel('Training samples')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('SVM_learning_curve.jpg')