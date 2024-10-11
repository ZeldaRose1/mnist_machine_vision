#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First attempt to make a machine vision model

Created on Tue Oct  8 05:48:22 2024

@author: zelda
"""

# Header: Imports
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
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

# Set hyperparameters for our RFC classifiers
n_est = 300  # Number of estimators
min_ss = 30  # Minimum samples split

# <codecell> Run original model and time it.

# Save start time
start_time1 = time.time()

# Initialize random forest classifier
rfc = RandomForestClassifier(n_estimators=n_est, min_samples_split=min_ss)

# Train the models
rfc.fit(X_train, y_train)

# Make predictions on train and test sets
train_pred = rfc.predict(X_train)
test_pred = rfc.predict(X_test)

# Compute accuracy scores
print("Accuracy score on non-transofrmed training set:",
      accuracy_score(y_train, train_pred))
print("Accuracy score on non-transformed testing set",
      accuracy_score(y_test, test_pred))

# Make predictions on prediction set.
pred_out = rfc.predict(pred)

# Save end time
end_time1 = time.time()
no_pca_rt = end_time1 - start_time1

# <codecell> Run PCA model and time it.

run_times = []
train_acc = []
test_acc = []
pcts = []

for pct in [x / 100 for x in range(5, 96, 5)]:
    # Save percents used
    pcts.append(pct)
    # Save start time
    start_time2 = time.time()

    # Train and transform data with principal component analysis
    pca = PCA(n_components=pct, whiten=True)
    X_train_pca = pca.fit(X_train).transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Initialize random forest classifier
    rfc_pca = RandomForestClassifier(n_estimators=n_est, min_samples_split=min_ss)
    
    # Train the models
    rfc_pca.fit(X_train_pca, y_train)
    
    # Make predictions on train and test sets
    train_pred_pca = rfc_pca.predict(X_train_pca)
    test_pred_pca = rfc_pca.predict(X_test_pca)
    
    # Compute accuracy scores
    # print("Accuracy score on pca-transofrmed training set:",
    #       accuracy_score(y_train, train_pred_pca))
    # print("Accuracy score on pca-transformed testing set",
    #       accuracy_score(y_test, test_pred_pca))
    train_acc.append(accuracy_score(y_train, train_pred_pca))
    test_acc.append(accuracy_score(y_test, test_pred_pca))
    
    # Make predictions on prediction set.
    pred_out_pca = rfc_pca.predict(pca.transform(pred))
    
    # Save end time
    end_time2 = time.time()
    pca_rt = end_time2 - start_time2
    run_times.append(pca_rt)
    
    print(f"Initial run time with {pct}% column reduction:\t\t", round(no_pca_rt, 2))
    print(f"PCA run time with {pct}% column reduction:\t\t", round(pca_rt), 2)

# <codecell> Plot the data from the PCA section.

# Combine lists into a single dataframe for ease of manipulation
results = pd.DataFrame({
    "run_time": run_times, "train_acc": train_acc, "test_acc": test_acc
    }, index = pcts)
# Should have saved the pct values instead of inputting them manually

# Label index
results.index.name = 'percent_reduction'
# Inspect dataframe matches expectations
print(results.sort_index())

# Plot results
# Form figure and subplot
fig, ax1 = plt.subplots()
# Clone subplot to plot two y axes on one x axis
ax2 = ax1.twinx()
# Form train and test lines
test_line = ax1.plot(results.index, results["test_acc"], color="red", marker='o', label='Test')
train_line = ax1.plot(results.index, results["train_acc"], color="blue", marker='o', label='Train')
time_line = ax2.plot(results.index, results["run_time"], color="yellow", marker='o', label='Time')
# Label Axes
ax1.set_ylabel("Accuracy")
ax2.set_ylabel("Run Time")
ax1.set_xlabel("Percent column reduction")
ax1.legend(labels=['Test', 'Train'])
plt.show()

# <codecell> Output final solution to a csv
# Uncomment everything to save.
# Convert to dataframe with proper format. Index + 1 is required by kaggle
out = pd.DataFrame(data=pred_out_pca, index=pred.index + 1, columns=['Label'])
out.index.name='ImageId'

# Save predictions as a .csv
out.to_csv("submission02.csv")
