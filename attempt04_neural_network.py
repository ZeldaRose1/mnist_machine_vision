#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trying to create a dense neural network with tensorflow to best classify
the mnist dataset.

Created on Tue Nov  5 05:59:51 2024

@author: zelda
"""
# <codecell> Imports and global parameters

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set random seed for all functions
SEED = 42

# Run tf eagerly
tf.config.run_functions_eagerly(True)

# <codecell> User defined functions


def calculate_accuracy(model, X, y):
    """
    Calculates accuracy of a model based on X for feature set
    and y for label set.

    Parameters:
        X: tensor constant of input features
        y: pd.DataFrame of correct labels corresponding to X

    Returns:
        accuracy score
    """
    # Make predictions based on the test set
    test_pred = model.predict(X)

    # Convert probabilities into predictions
    pred_list = []
    for i in range(len(test_pred)):
        col_max = test_pred[i].max()
        for j in range(test_pred.shape[1]):
            if test_pred[i][j] == col_max:
                pred_list.append(j)

    # Make new dataframe from y_test and test_pred to calculate accuracy
    result_df = pd.DataFrame(zip(y, pred_list), columns=['real', 'predicted'])

    # Check how many predictions match
    result_df['match'] = result_df.apply(
        lambda x: 1 if x['real'] == x['predicted'] else 0,
        axis=1
    )

    # Calculate accuracy
    print("Model accuracy is ", str(result_df['match'].sum() / len(result_df)))
    return result_df['match'].sum() / len(result_df)


# <codecell> Import data
df = pd.read_csv("train.csv")
pred = pd.read_csv("test.csv")

# Split data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis='columns'),
    df['label'],
    test_size=0.25,
    random_state=SEED
)

# Ensure inputs are tensors not pandas dataframes
X_train_tensor = tf.constant(X_train, tf.float32)
X_test_tensor = tf.constant(X_test, tf.float32)

# Run a one_hot encoder on y_train
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=10)

# <codecell> Define and compile model

# Define input layer.
inputs = tf.keras.Input(shape=(X_train.shape[1],))

# Define first dense layer
layer1 = tf.keras.layers.Dense(512, activation='sigmoid')(inputs)
layer2 = tf.keras.layers.Dense(256, activation='sigmoid')(layer1)

# Define dropout layer
dropout1 = tf.keras.layers.Dropout(0.25)(layer2)

# Define output layer
output = tf.keras.layers.Dense(10, activation='softmax')(dropout1)

# Define model
model = tf.keras.Model(inputs=inputs, outputs=output)

# Compile model with optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fit model
model.fit(X_train_tensor, y_train_oh, epochs=10)

calculate_accuracy(model, X_test_tensor, y_test)
