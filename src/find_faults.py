"""
find_faults.py

This module contains functions for identifying occurrences in a description field. It uses a trained
classifier model to predict whether a description contains an occurrence or not. Please note that the 
code and data in this repository have been modified for confidentiality purposes, preserving the sensitive 
details of the project that generated the article. As a result, some functions may appear more generic, 
but they are fully functional and demonstrate all the concepts discussed in the article.

Functions:
- load_model(model_path): Loads the trained classifier model from a file.
- identify_occurrences(descriptions, model): Identifies occurrences in a list of descriptions using a 
  trained model.
- run_identification(data, description_column, model_path): Runs the identification process on a dataset.

Example Usage:
data = load_csv('data.csv')
model = load_model('classifier_model.pkl')
identified_occurrences = identify_occurrences(data['description'], model)

Authors: This code was developed by the Data Science Laboratory at the Polytechnic School 
of the University of São Paulo, led by researchers Wellingthon Queiroz and Osvaldo Gogliano, 
as described in the article.
         Main contacts:
            - Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
            - Osvaldo Gogliano - ogogli@gmail.com
version: 0.15rc

"""

import pandas as pd
import pickle
from data_loader import load_csv, load_pickle, load_postgres, load_xls
from train_test import get_model

def load_model(model_path=None, manual=False, data=None):
    """
    If manual == True, Loads the trained classifier model from a file.
    otherwise it will use our model from the train_test.py module

    Args:
        model_path (str): File path of the trained model.

    Returns:
        object: Loaded classifier model.

    """
    if manual:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = get_model(data, 'target')
    
    return model

def identify_occurrences(descriptions, model):
    """
    Identifies occurrences in a list of descriptions using a trained model.

    Args:
        descriptions (list): List of descriptions to be analyzed.
        model (object): Trained classifier model.

    Returns:
        pd.DataFrame: DataFrame with the original descriptions and the corresponding occurrence predictions.

    """
    occurrences = model.predict(descriptions)
    identified_occurrences = pd.DataFrame({'description': descriptions, 'occurrence': occurrences})
    return identified_occurrences

def run_identification(data, description_column, model_path):
    """
    Runs the identification process on a dataset.

    Args:
        data (pd.DataFrame): Dataset containing the descriptions.
        description_column (str): Name of the column containing the descriptions.
        model_path (str): File path of the trained model.

    Returns:
        pd.DataFrame: DataFrame with the original descriptions and the corresponding occurrence predictions.

    """
    descriptions = data[description_column]
    model = load_model(model_path)
    identified_occurrences = identify_occurrences(descriptions, model)
    return identified_occurrences


# Example usage
data_csv = load_csv('data.csv')
data_pickle = load_pickle('data.pickle')
data_xls = load_xls('data.xls')
data_postgres = load_postgres('postgres_config.json')

identified_occurrences = run_identification(data_pickle, 'description', 'classifier_model.pkl')
print(identified_occurrences.head())
