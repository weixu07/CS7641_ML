import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

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

dtree = DecisionTreeClassifier(max_depth=1, min_samples_leaf = 1)
dtree_Ada = AdaBoostClassifier(base_estimator=dtree, random_state=100)
dtree_Ada.fit(X_train, y_train)
predictions = dtree_Ada.predict(X_test)
print("Boosting [Ada]---> outputs")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Validation Curve
learners_range = np.arange(1, 200)
train_scores, test_scores = validation_curve(dtree_Ada, X_train, y_train, param_name="n_estimators",param_range=learners_range, cv=5)

plt.figure()
plt.plot(learners_range, np.mean(train_scores, axis=1), label='Training')
plt.plot(learners_range, np.mean(test_scores, axis=1), label='Testing')
plt.title('Validation for Boosting')
plt.xlabel('Number of Learners')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Boosting_validation_curve_estimators.jpg')


'''
lr_range = np.logspace(-5, 0, 6)
train_scores, test_scores = validation_curve(dtree_Ada, X_train, y_train, param_name="learning_rate", param_range=lr_range, cv=5)
plt.figure()
plt.semilogx(lr_range, np.mean(train_scores, axis=1), label='Training')
plt.semilogx(lr_range, np.mean(test_scores, axis=1), label='Testing')
plt.title('Validation for Boosting')
plt.xlabel('Learning Rate')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Boosting_validation_curve_LearnRate.jpg')
'''

# WITH n_estimators = 1
dtree_Ada_best = AdaBoostClassifier(base_estimator=dtree, random_state=100, n_estimators=1)
dtree_Ada_best.fit(X_train,y_train)
predictions_best = dtree_Ada_best.predict(X_test)
print("Ada [n_estimators = 1]---> outputs")
print(classification_report(y_test, predictions_best))
print(confusion_matrix(y_test, predictions_best))

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(dtree_Ada_best, X_train, y_train, train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Testing')
plt.title('Learning curve for Boosting')
plt.xlabel('Fraction of training examples')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Booting_learning_curve.jpg')


