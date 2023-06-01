"""
match_ocorrencias.py

This module contains functions for matching the identified occurrences with the data from the instrumented wagon. 
It performs the matching based on common fields, such as longitude, latitude, and timestamp. Please note that the 
code and data in this repository have been modified for confidentiality purposes, preserving the sensitive 
details of the project that generated the article. As a result, some functions may appear more generic, 
but they are fully functional and demonstrate all the concepts discussed in the article.

Functions:
- match_occurrences_with_data(occurrences, instrumented_data): Matches the identified occurrences with the instrumented data.
- run_matching(occurrence_file, instrumented_data_file, output_file): Runs the matching process on the occurrence file
  and the instrumented data file.

Example Usage:
run_matching('identified_occurrences.csv', 'instrumented_data.csv', 'matched_data.csv')

Authors: This code was developed by the Data Science Laboratory at the Polytechnic School 
of the University of São Paulo, led by researchers Wellingthon Queiroz and Osvaldo Gogliano, 
as described in the article.
         Main contacts:
            - Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
            - Osvaldo Gogliano - ogogli@gmail.com
version: 0.15rc

"""

import pandas as pd
from find_faults import identify_occurrences

def match_occurrences_with_data(occurrences, instrumented_data):
    """
    Matches the identified occurrences with the instrumented data based on common fields.

    Args:
        occurrences (pd.DataFrame): DataFrame containing the identified occurrences.
        instrumented_data (pd.DataFrame): DataFrame containing the instrumented data.

    Returns:
        pd.DataFrame: DataFrame with the matched occurrences and instrumented data.

    """
    matched_data = pd.merge(occurrences, instrumented_data, on=['longitude', 'latitude', 'timestamp'], how='inner')
   
    return matched_data

def run_matching(occurrence_file, instrumented_data_file, output_file):
    """
    Runs the matching process on the occurrence file and the instrumented data file.

    Args:
        occurrence_file (str): File path of the occurrence file.
        instrumented_data_file (str): File path of the instrumented data file.
        output_file (str): File path for saving the matched data.

    """
    occurrences = identify_occurrences('faults_dataset.csv')
    instrumented_data = pd.read_csv(instrumented_data_file)
    matched_data = match_occurrences_with_data(occurrences, instrumented_data)
    matched_data.to_csv(output_file, index=False)

# Example usage
run_matching('identified_occurrences.csv', 'instrumented_data.csv', 'matched_data.csv')
