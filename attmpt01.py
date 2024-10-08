#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First attempt to make a machine vision model

Created on Tue Oct  8 05:48:22 2024

@author: zelda
"""

# Header: Imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  #, KFold, GridSearchCV

# Set random seed
SEED = 42

# <codecell> Import data and opening data visualization.
df = pd.read_csv("train.csv")
pred = pd.read_csv("test.csv")

# Split data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis='columns'),
    df['label'],
    test_size=0.25,
    random_state=SEED
)

# =============================================================================
# This block of code does not run quickly. Couldn't wait for it to finish running
# Going with arbitrary hyperparameters
# # Initialize a 5-fold cross validation search
# cv = KFold(n_splits=5, shuffle=True)
# 
# # Initialize param grid
# param_grid = [
#     {
#          "n_estimators": [100, 200, 300],
#          "min_samples_split": [10, 30, 50]
#      }
# ]
# 
# # Initialize grid search model
# clf = GridSearchCV(
#     estimator=RandomForestClassifier(),
#     param_grid=param_grid,
#     cv=cv
# )
# 
# # Fit grid search
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# =============================================================================

# Initialize random forest classifier
rfc = RandomForestClassifier(n_estimators=300, min_samples_split=30)

# Train the model
rfc.fit(X_train, y_train)

# Make predictions on train and test sets
train_pred = rfc.predict(X_train)
test_pred = rfc.predict(X_test)

# Compute accuracy scores
accuracy_score(y_train, train_pred)
accuracy_score(y_test, test_pred)

# Make predictions on prediction set.
pred_out = rfc.predict(pred)
out = pd.DataFrame(data=pred_out, index=pred.index, columns=['Label'])
out.reset_index(replace=True)
