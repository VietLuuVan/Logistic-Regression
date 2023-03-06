# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:58:58 2023

@author: thoidaipc
"""
from pandas import *
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Reading CSV file
data = read_csv("Admission_Predict.csv")
X = data.iloc[:,1:8]
y = data.iloc[:,8]

# Convert continuous variable to discrete variable
y_labels = [1 if i > 0.75 else 0 for i in y]        
y_labels = np.array(y_labels)

# Preparing train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, train_size = 350, shuffle = False)
Xbar = np.array(np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1))

# Training model
logReg = LogisticRegression(max_iter=350)
logReg.fit(Xbar, y_train)
print(logReg.coef_)

# Running model on test set
Xtest = np.array(np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1))
y_pred = logReg.predict(Xtest)

# Evaluating model using Accuracy, Precision and Recall
print("Accuracy = {}".format(accuracy_score(y_test, y_pred)))
print("Precision = {}".format( precision_score(y_test, y_pred)))
print("Recall = {}".format( recall_score(y_test, y_pred)))