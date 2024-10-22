#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 06:43:15 2024

@author: zelda
"""


# Header: Imports
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC

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

# <codecell> Run original model and time it.

# Save start time
start_time1 = time.time()

# Initialize random forest classifier
svc = SVC(kernel='rbf', random_state=SEED)

# Train the models
svc.fit(X_train, y_train)

# Make predictions on train and test sets
train_pred = svc.predict(X_train)
test_pred = svc.predict(X_test)

# Compute accuracy scores
print("Accuracy score on non-transofrmed training set:",
      accuracy_score(y_train, train_pred))
print("Accuracy score on non-transformed testing set",
      accuracy_score(y_test, test_pred))

# Make predictions on prediction set.
pred_out = svc.predict(pred)

# Save end time
end_time1 = time.time()
no_pca_rt = end_time1 - start_time1
print(no_pca_rt)

# <codecell> Run model with PCA and time it.

# Save start time
start_time1 = time.time()

# Initialize PCA model
pca = PCA(n_components=0.60, whiten=True)

# Train and transform data
X_train_pca = pca.fit(X_train).transform(X_train)
X_test_pca = pca.transform(X_test)
pred_pca = pca.transform(pred)

# Initialize random forest classifier
svc = SVC(random_state=SEED)

# Train the models
svc.fit(X_train_pca, y_train)

# Make predictions on train and test sets
train_pred_pca = svc.predict(X_train_pca)
test_pred_pca = svc.predict(X_test_pca)

# Compute accuracy scores
print("Accuracy score on transofrmed training set:",
      accuracy_score(y_train, train_pred_pca))
print("Accuracy score on transformed testing set",
      accuracy_score(y_test, test_pred_pca))

# Make predictions on prediction set.
# pred_out = mlp.predict(pred)

# Save end time
end_time1 = time.time()
no_pca_rt = end_time1 - start_time1
print(no_pca_rt)

# =============================================================================
# # <codecell> Run PCA model and time it.
# 
# run_times = []
# train_acc = []
# test_acc = []
# pcts = []
# 
# for pct in [x / 100 for x in range(5, 96, 5)]:
#     # Save percents used
#     pcts.append(pct)
#     # Save start time
#     start_time2 = time.time()
# 
#     # Train and transform data with principal component analysis
#     pca = PCA(n_components=pct, whiten=True)
#     X_train_pca = pca.fit(X_train).transform(X_train)
#     X_test_pca = pca.transform(X_test)
#     
#     # Initialize random forest classifier
#     mlp_pca = MLPClassifier(random_state=SEED)
#     
#     # Train the models
#     mlp_pca.fit(X_train_pca, y_train)
#     
#     # Make predictions on train and test sets
#     train_pred_pca = mlp_pca.predict(X_train_pca)
#     test_pred_pca = mlp_pca.predict(X_test_pca)
#     
#     # Compute accuracy scores
#     # print("Accuracy score on pca-transofrmed training set:",
#     #       accuracy_score(y_train, train_pred_pca))
#     # print("Accuracy score on pca-transformed testing set",
#     #       accuracy_score(y_test, test_pred_pca))
#     train_acc.append(accuracy_score(y_train, train_pred_pca))
#     test_acc.append(accuracy_score(y_test, test_pred_pca))
#     
#     # Make predictions on prediction set.
#     pred_out_pca = mlp_pca.predict(pca.transform(pred))
#     
#     # Save end time
#     end_time2 = time.time()
#     pca_rt = end_time2 - start_time2
#     run_times.append(pca_rt)
#     
#     print(f"Initial run time with {pct}% column reduction:\t\t", round(no_pca_rt, 2))
#     print(f"PCA run time with {pct}% column reduction:\t\t", round(pca_rt), 2)
# 
# # <codecell> Plot the data from the PCA section.
# 
# # Combine lists into a single dataframe for ease of manipulation
# results = pd.DataFrame({
#     "run_time": run_times, "train_acc": train_acc, "test_acc": test_acc
#     }, index = pcts)
# # Should have saved the pct values instead of inputting them manually
# 
# # Label index
# results.index.name = 'percent_reduction'
# # Inspect dataframe matches expectations
# print(results)
# 
# # Plot results
# # Form figure and subplot
# fig, ax1 = plt.subplots()
# # Clone subplot to plot two y axes on one x axis
# ax2 = ax1.twinx()
# # Form train and test lines
# test_line = ax1.plot(results.index, results["test_acc"], color="red", marker='o', label='Test')
# train_line = ax1.plot(results.index, results["train_acc"], color="blue", marker='o', label='Train')
# time_line = ax2.plot(results.index, results["run_time"], color="black", marker='o', label='Time')
# # Label Axes
# ax1.set_ylabel("Accuracy")
# ax2.set_ylabel("Run Time")
# ax1.set_xlabel("Percent column reduction")
# # Must add the lines together for the legend with multiple axes
# lines = test_line + train_line + time_line
# labels = [line.get_label() for line in lines]
# ax1.legend(lines, labels, loc=0)
# # Set title
# plt.title("Accuracy and runtime vs pct column reduction")
# 
# plt.show()
# 
# =============================================================================
# <codecell> Found ideal pct reduction. Transform variables

# Initialize PCA model
pca = PCA(n_components=0.55, whiten=True)

# Train and transform data
X_train_pca = pca.fit(X_train).transform(X_train)
X_test_pca = pca.transform(X_test)
pred_pca = pca.transform(pred)

# <codecell> Retrain model without splitting training and test data
# Reload data
X_final = df.drop("label", axis=1)
y_final = df['label']

# Transform data
# X_final = pca.fit(X_final).transform(X_final)
# pred_final = pca.transform(pred)

# Initialize and retrain model
svc = SVC(random_state=SEED)
svc.fit(X_final, y_final)

# Make final predictions
out = svc.predict(pred)
# <codecell> Output final solution to a csv
# Uncomment everything to save.
# Convert to dataframe with proper format. Index + 1 is required by kaggle
out_f = pd.DataFrame(data=out, index=pred.index + 1, columns=['Label'])
out_f.index.name='ImageId'

# Save predictions as a .csv
out_f.to_csv("submission04.csv")
