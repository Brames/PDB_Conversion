"""
============================
Gradient Boosting regression
============================

Gradient Boosting on the 1322 protein dataset features.

This example fits a Gradient Boosting model with least squares loss and
500 regression trees of depth 4.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn import cross_validation
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier #For Classification

################################################################################
print("Load data")
protein = np.loadtxt('All_Features.txt')
protein = np.nan_to_num(protein)
#protein = scale(protein, axis=0, with_mean=True, with_std=True, copy=True )
protein = protein.T
target  = np.loadtxt('binding_FE.txt')

# print(protein.shape,target.shape)
X, y = shuffle(protein,target, random_state=425)

# X = X.astype(np.float32)
# offset = int(X.shape[0] * 0.9)
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=425)

#################################################################################
print("Fit Regression Model")
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
          
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

#################################################################################
# Cross Validation
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(protein, target, test_size=0.4, random_state=0)
#print("Cross Validation: %.4f" % clf.score(X_test,y_test))
#scores = cross_validation.cross_val_score(clf,protein,target,cv=5)
#print(scores)


#################################################################################
print("Perform Cross Validation")
##scores = cross_validation.cross_val_score(clf, protein, target, cv=5)
scores = cross_validation.cross_val_score(clf, protein, target, scoring='mean_squared_error', cv=5)
print(scores)
#print(scores.mean())

#################################################################################
#print("Plot Training Deviance")
#
## compute test set deviance
#test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#
#for i, y_pred in enumerate(clf.staged_predict(X_test)):
#    test_score[i] = clf.loss_(y_test, y_pred)
#
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title('Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#         label='Training Set Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#         label='Test Set Deviance')
#plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
#plt.ylabel('Deviance')
#plt.show()

###############################################################################
