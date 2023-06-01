"""
classifier.py

This module contains functions for training and evaluating a classifier on preprocessed text data.
It includes functionality for loading data, preprocessing, splitting into train and test sets, training a classifier,
and evaluating the accuracy of the classifier.

Please note that the code and data in this repository have been modified for confidentiality purposes, preserving the
sensitive details of the project that generated the article. As a result, some functions may appear more generic,
but they are fully functional and demonstrate all the concepts discussed in the article.

Functions:
- train_classifier(X_train, y_train): Trains a classifier on the given training data.
- evaluate_classifier(model, X_test, y_test): Evaluates the accuracy of the trained classifier on the given test data.
- run_classification(input_file): Loads the dataset, preprocesses the data, splits into train and test sets,
  trains a classifier, and evaluates its accuracy.

Example Usage:
run_classification('data.csv')

Authors: This code was developed by the Data Science Laboratory at the Polytechnic School
of the University of São Paulo, led by researchers Wellingthon Queiroz and Osvaldo Gogliano,
as described in the article.
         Main contacts:
            - Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
            - Osvaldo Gogliano - ogogli@gmail.com
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging
from data_loader import load_csv, load_pickle, load_postgres, load_xls
from preprocessing import preprocess_data
from train_test import split_train_test_data

logging.basicConfig(filename='classifier.log', level=logging.INFO)

def train_classifier(X_train, y_train):
    # Initialize the classifier
    model = LogisticRegression()

    # Training steps
    model.fit(X_train, y_train)

    return model

def evaluate_classifier(model, X_test, y_test):
    # Evaluation steps
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def run_classification(input_file):
    """
    Loads the dataset, preprocesses the data, splits into train and test sets,
    trains a classifier, and evaluates its accuracy.

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

# Example usage
data_csv = load_csv('data.csv')
data_pickle = load_pickle('data.pickle')
data_xls = load_xls('data.xls')
data_postgres = load_postgres('postgres_config.json')

run_classification(data_pickle)
