#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:18:26 2017

@author: Vishnusankar
"""


from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from HelperClass import HelperClass

color = sns.color_palette()


pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns',500)

originalTrainData = pd.read_csv('./train_u6lujuX_CVtuZ9i.csv')
trainData = originalTrainData 
trainData.shape

trainData.head()

# Missing Value
miss_data = trainData.isnull().sum(axis=0)
print(miss_data)

#------------------------------------------------ X_train Data ----------------------------------------------------------------#

#---------------- Fill Mising Values--------------------------------#
## Gender Column
trainData['Gender'].fillna('Other', inplace = True)

## Married Column
trainData['Married'].fillna('Yes', inplace = True)

## Dependents Column
trainData['Dependents'].fillna('4', inplace = True)

## Self_Employed Column
trainData['Self_Employed'].fillna('Yes', inplace = True)

## LoanAmount Column
trainData['LoanAmount'].fillna((trainData['LoanAmount'].mean()), inplace = True)

## Loan_Amount_Term Column
trainData['Loan_Amount_Term'].fillna((trainData['Loan_Amount_Term'].mean()), inplace = True)

## Credit_History Column
trainData['Credit_History'].fillna((trainData['Credit_History'].mean()), inplace = True)

withOutLoanIDValuetrainData = trainData.drop('Loan_ID', 1)
withOutTargetValuetrainData = withOutLoanIDValuetrainData.drop('Loan_Status', 1)
#withOutTargetValuetrainData= pd.get_dummies(withOutTargetValuetrainData)

#---------------- Fill Mising Values--------------------------------

# Label Encoding categorical data
X = withOutTargetValuetrainData.iloc[:,0:11].values
y = trainData.iloc[:, 12].values

loadAmout_loadAmoutTerm = X[:,8:10]
np.cov(loadAmout_loadAmoutTerm.astype(float), rowvar=False)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from HelperClass import HelperClass
#
labelencoder_X_2 = LabelEncoder()
#
# Gender Column
X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0])

# Married Column
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])

# Dependents Column
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Self_Employed Column
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])

# Education Column
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])

# Property Area Column
X[:, 10] = labelencoder_X_2.fit_transform(X[:, 10])

## Dummy Variables
#
# Load Status Column
y = labelencoder_X_2.fit_transform(y)

## Gender Column
#oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[0])
#X = oneHotEncoder_X_2.fit_transform(X).toarray()
##
##
### Dependents Column
#oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[4])
#X = oneHotEncoder_X_2.fit_transform(X).toarray()
##
## Area Column
#oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[16])
#X = oneHotEncoder_X_2.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Feature Extraction

# Feature Selection
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score



seed = 7
kfold = model_selection.KFold(n_splits = 10, random_state = seed)
cart = XGBClassifier()

#estimators = []
#model1 = ExtraTreesClassifier()
#estimators.append(('ExtraTreesClassifier', model1))
#model2 = XGBClassifier()
#estimators.append(('XGBClassifier', model2))
#model3 = RandomForestClassifier()
#estimators.append(('RandomForestClassifier', model3))

ensemble = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=seed)
#estimators = []
#model2 = DecisionTreeClassifier()
#estimators.append(('cart', model2))
#model3 = SVC()
#estimators.append(('svm', model3))
#model4 = XGBClassifier()
#estimators.append(('XGB', model3))
## create the ensemble model
#ensemble = VotingClassifier(estimators)

results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)
print(results.mean())

#0.79318507891 = 1 n_splits, 50 n_estimators, with hotEncoder
#0.796562665256 = 10 n_splits, 50 n_estimators, with hotEncoder
#0.803067160233 = 10 n_splits, 100 n_estimators, with hotEncoder, XGBClassifier
#0.803093601269 = 10 n_splits, 100 n_estimators, disable hotEncoder, XGBClassifier
#0.791530944625 = 1 n_splits, 100 n_estimators, with hotEncoder
#0.791530944625 = 10 n_splits, 130 n_estimators, with hotEncoder
#0.767345319937 = 10 n_splits, 130 n_estimators, disable hotEncoder
#0.773849814913 = 10 n_splits, 130 n_estimators, with hotEncoder
ensemble.fit(X,y)
y_pred = ensemble.predict(X)


predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#------------------------------------------------ X_train Data ----------------------------------------------------------------#

#------------------------------------------------ X_test Data ----------------------------------------------------------------#

#---------------- Fill Mising Values--------------------------------#
originalTestData = pd.read_csv('./test_Y3wMUE5_7gLdaTN.csv')
# Missing Value
miss_data = originalTestData.isnull().sum(axis=0)
print(miss_data)

## Gender Column
originalTestData['Gender'].fillna('Other', inplace = True)

## Dependents Column
originalTestData['Dependents'].fillna('4', inplace = True)

## Self_Employed Column
originalTestData['Self_Employed'].fillna('Yes', inplace = True)

## LoanAmount Column
originalTestData['LoanAmount'].fillna((originalTestData['LoanAmount'].mean()), inplace = True)

## Loan_Amount_Term Column
originalTestData['Loan_Amount_Term'].fillna((originalTestData['Loan_Amount_Term'].mean()), inplace = True)

## Credit_History Column
originalTestData['Credit_History'].fillna((originalTestData['Credit_History'].mean()), inplace = True)

withOutLoadIDTestData = originalTestData.drop('Loan_ID', 1)
#withOutLoadIDTestData = pd.get_dummies(withOutLoadIDTestData)

#---------------- Fill Mising Values--------------------------------

# Label Encoding categorical data
X_test = withOutLoadIDTestData .iloc[:,0:24].values

# Gender Column
X_test[:, 0] = labelencoder_X_2.fit_transform(X_test[:, 0])

# Married Column
X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])

# Dependents Column
X_test[:, 2] = labelencoder_X_2.fit_transform(X_test[:, 2])

# EducationColumn
X_test[:, 3] = labelencoder_X_2.fit_transform(X_test[:, 3])

# Self_Employed  Column
X_test[:, 4] = labelencoder_X_2.fit_transform(X_test[:, 4])

# Property Area Column
X_test[:, 10] = labelencoder_X_2.fit_transform(X_test[:, 10])


# Dummy Variables

## Gender Column
#oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[0])
#X_test = oneHotEncoder_X_2.fit_transform(X_test).toarray()
#
#
## Gender Column
#oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[4])
#X_test = oneHotEncoder_X_2.fit_transform(X_test).toarray()
#
## Gender Column
#oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[16])
#X_test = oneHotEncoder_X_2.fit_transform(X_test).toarray()

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_test = sc.fit_transform(X_test)

#y_pred_Train = classifier.predict_on_bestEstimator(X_test,'RandomForestClassifier')
#y_pred_Train = classifier.predict(X_test)

y_pred_Train = ensemble.predict(X_test)


y_pred_Train = ["Y" if i == 1 else "N" for i in y_pred_Train]
pd.DataFrame({"Loan_ID" : originalTestData.Loan_ID, "Loan_Status" : y_pred_Train}).to_csv('sample_submission.csv', index=False)
#result = model.score_summary(sort_by='min_score')

