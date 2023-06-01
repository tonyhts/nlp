"""
train_test.py

This module contains functions for training and evaluating machine learning models using different
algorithms and data splitting strategies. Please note that the code and data in this repository 
have been modified for confidentiality purposes, preserving the sensitive details of the project 
that generated the article. As a result, some functions may appear more generic, but they are fully 
functional and demonstrate all the concepts discussed in the article.

Functions:
- split_data(X, y, test_size=0.2, random_state=None): Splits the data into training and testing sets.
- train_model(X_train, y_train, model): Trains the machine learning model using the training data.
- evaluate_model(X_test, y_test, model): Evaluates the machine learning model using the testing data.
- run_experiment(data, target_column, model): Runs a complete machine learning experiment, including data 
  splitting, model training, and evaluation.

Example Usage:
data = load_csv('data.csv')
run_experiment(data, 'target', LogisticRegression())

Authors: This code was developed by the Data Science Laboratory at the Polytechnic School 
of the University of São Paulo, led by researchers Wellingthon Queiroz and Osvaldo Gogliano, 
as described in the article.
         Main contacts:
            - Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
            - Osvaldo Gogliano - ogogli@gmail.com

version: 0.15rc

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import load_csv, load_pickle, load_postgres, load_xls

def split_data(X, y, test_size=0.2, random_state=None):
    """
    Splits the data into training and testing sets.

    Args:
        X (pd.DataFrame): Features data.
        y (pd.Series): Target data.
        test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to None.

    Returns:
        tuple: Tuple containing the split data: X_train, X_test, y_train, y_test.

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model):
    """
    Trains the machine learning model using the training data.

    Args:
        X_train (pd.DataFrame): Features data for training.
        y_train (pd.Series): Target data for training.
        model (object): Machine learning model.

    Returns:
        object: Trained machine learning model.

    """
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(X_test, y_test, model):
    """
    Evaluates the machine learning model using the testing data.

    Args:
        X_test (pd.DataFrame): Features data for testing.
        y_test (pd.Series): Target data for testing.
        model (object): Trained machine learning model.

    Returns:
        float: Accuracy score of the model.

    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
   
    return accuracy

def run_experiment(data, target_column, model, test_size=0.2, random_state=None):
    """
    Runs a complete machine learning experiment, including data splitting, model training, and evaluation.

    Args:
        data (pd.DataFrame): Complete dataset.
        target_column (str): Name of the target column.
        model (object): Machine learning model to be used.
        test_size (float, optional): Proportion of the data to be used for testing. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to None.

    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    trained_model = train_model(X_train, y_train, model)
    accuracy = evaluate_model(X_test, y_test, trained_model)

    print('Accuracy:', accuracy)
    return trained_model

def get_model(data, target):
    model = run_experiment(data, target, LogisticRegression())
    return model

# Example usage
data_csv = load_csv('data.csv')
data_pickle = load_pickle('data.pickle')
data_xls = load_xls('data.xls')
data_postgres = load_postgres('postgres_config.json')

get_model(data_pickle, 'target')

