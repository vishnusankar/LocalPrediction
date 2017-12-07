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

#---------------- Fill Mising Values--------------------------------

# Label Encoding categorical data
X = trainData.iloc[:,1:12].values
y = trainData.iloc[:, 12].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from HelperClass import HelperClass

labelencoder_X_2 = LabelEncoder()

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

# Dummy Variables

# Load Status Column
y = labelencoder_X_2.fit_transform(y)

# Gender Column
oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_X_2.fit_transform(X).toarray()


# Gender Column
oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[4])
X = oneHotEncoder_X_2.fit_transform(X).toarray()


oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[16])
X = oneHotEncoder_X_2.fit_transform(X).toarray()


#OneHotEncoder_X_1 = OneHotEncoder(categorical_features = [0])
#pd.Series(np.where(lambda x: dict(yes=1, no=0)[x],
#              X.tolist()),X.index)
#X = OneHotEncoder_X_1.fit_transform(X[:1]).toarray()

#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]


# Feature Extraction

# Feature Selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.ensemble import RandomForestClassifier
from  EstimatorSelectionHelper import EstimatorSelectionHelper
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

models = {
            'RandomForestClassifier' : RandomForestClassifier(),
            'XGBClassifier' : XGBClassifier()
            }
    
parameter = {
            'RandomForestClassifier' : { 'max_depth' : [2,5,7,9],
                                        'n_estimators': [200, 700],
                                        'max_features': ['auto', 'sqrt', 'log2'],
                                        'criterion' : ['gini', 'entropy']},
            'XGBClassifier' : {}
            }
    
classifier = EstimatorSelectionHelper(models, parameter)
classifier.fit(X, y, scoring='f1', cv = 3, n_jobs=1,refit=True,verbose=2)
y_pred = classifier.predict(X_test)


classifier = XGBClassifier()
classifier.fit(X,y)
y_pred = classifier.predict_on_bestEstimator(X_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
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

#---------------- Fill Mising Values--------------------------------

# Label Encoding categorical data
X_test = originalTestData.iloc[:,1:12].values

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

# Gender Column
oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[0])
X_test = oneHotEncoder_X_2.fit_transform(X_test).toarray()


# Gender Column
oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[4])
X_test = oneHotEncoder_X_2.fit_transform(X_test).toarray()

# Gender Column
oneHotEncoder_X_2 = OneHotEncoder(categorical_features=[16])
X_test = oneHotEncoder_X_2.fit_transform(X_test).toarray()

#y_pred_Train = classifier.predict_on_bestEstimator(X_test,'RandomForestClassifier')

y_pred_Train = classifier.predict(X_test)



y_pred_Train = ["Y" if i == 1 else "N" for i in y_pred_Train]
pd.DataFrame({"Loan_ID" : originalTestData.Loan_ID, "Loan_Status" : y_pred_Train}).to_csv('sample_submission.csv', index=False)
result = classifier.score_summary(sort_by='min_score')

