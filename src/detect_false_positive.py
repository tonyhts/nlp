"""
detect_false_positive.py

This module contains functions for detecting false positives in the classification results.
It includes functionality for loading data, preprocessing, splitting into train and test sets,
training a classifier, evaluating its accuracy, and detecting false positives using a confusion matrix.

Please note that the code and data in this repository have been modified for confidentiality purposes, preserving the
sensitive details of the project that generated the article. As a result, some functions may appear more generic,
but they are fully functional and demonstrate all the concepts discussed in the article.

Functions:
- detect_false_positives(X_test, y_test, model): Detects false positives using a confusion matrix.
- run_false_positive_detection(input_file): Loads the dataset, preprocesses the data, splits into train and test sets,
  trains a classifier, evaluates its accuracy, and detects false positives.

Example Usage:
run_false_positive_detection('data.csv')

Authors: This code was developed by the Data Science Laboratory at the Polytechnic School
of the University of São Paulo, led by researchers Wellingthon Queiroz and Osvaldo Gogliano,
as described in the article.
         Main contacts:
            - Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
            - Osvaldo Gogliano - ogogli@gmail.com
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import logging
from data_loader import load_csv, load_xls, load_postgres, load_pickle
from preprocessing import preprocess_data
from train_test import split_train_test_data
from classifier import train_classifier, evaluate_classifier

logging.basicConfig(filename='false_positive.log', level=logging.INFO)

def detect_false_positives(X_test, y_test, model):
    # Predict labels for test data
    y_pred = model.predict(X_test)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # False positive detection logic
    false_positives = []
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 1:
            false_positives.append(X_test[i])

    return false_positives, cm

def run_false_positive_detection(input_file):
    """
    Loads the dataset, preprocesses the data, splits into train and test sets,
    trains a classifier, evaluates its accuracy, and detects false positives.

    Args:
        input_file (str): File path of the input dataset.

    """
    # Load the dataset
    data = load_csv(input_file)
    if data is None:
        return

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Split the data into features and labels
    X = preprocessed_data['features']
    y = preprocessed_data['labels']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test_data(X, y)

    # Train the classifier
    model = train_classifier(X_train, y_train)

    # Evaluate the classifier
    accuracy = evaluate_classifier(model, X_test, y_test)
    logging.info(f"Accuracy: {accuracy}")

    # Detect false positives
    false_positives, cm = detect_false_positives(X_test, y_test, model)
    logging.info(f"False Positives: {false_positives}")
    logging.info(f"Confusion Matrix:\n{cm}")


# Example usage
data_csv = load_csv('data.csv')
data_pickle = load_pickle('data.pickle')
data_xls = load_xls('data.xls')
data_postgres = load_postgres('postgres_config.json')


run_false_positive_detection(data_pickle)
