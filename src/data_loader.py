"""
data_loader.py

This module contains functions for loading data from different sources, 
including CSV files, Pickle files, XLS files, and PostgreSQL databases.
Please note that the code and data in this repository have been modified 
for confidentiality purposes, preserving the sensitive details of the project 
that generated the article.
As a result, some functions may appear more generic, but they are fully functional 
and demonstrate all the concepts discussed in the article.

Functions:
- load_csv(file_path): Loads data from a CSV file.
- load_pickle(file_path): Loads data from a Pickle file.
- load_xls(file_path): Loads data from an XLS file.
- load_postgres(config_file): Loads data from a PostgreSQL database.

Example Usage:
data_csv = load_csv('data.csv')
data_pickle = load_pickle('data.pickle')
data_xls = load_xls('data.xls')
data_postgres = load_postgres('postgres_config.json')

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
import psycopg2
import json
import logging


def load_csv(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: The loaded data as a pandas DataFrame, or None if the file is not found or an error occurs.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("CSV file loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("CSV file not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading CSV file: {str(e)}")
        return None


def load_pickle(file_path):
    """
    Load data from a Pickle file.

    Args:
        file_path (str): The path to the Pickle file.

    Returns:
        object or None: The loaded data object, or None if the file is not found or an error occurs.
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        logging.info("Pickle file loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("Pickle file not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading Pickle file: {str(e)}")
        return None


def load_xls(file_path):
    """
    Load data from an XLS file.

    Args:
        file_path (str): The path to the XLS file.

    Returns:
        pd.DataFrame or None: The loaded data as a pandas DataFrame, or None if the file is not found or an error occurs.
    """
    try:
        data = pd.read_excel(file_path)
        logging.info("XLS file loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("XLS file not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading XLS file: {str(e)}")
        return None


def load_postgres(config_file):
    """
    Load data from a PostgreSQL database.

    Args:
        config_file (str): The path to the PostgreSQL configuration file.

    Returns:
        list or None: The loaded data as a list of tuples, or None if the configuration file is not found or an error occurs.
    """
    try:
        with open(config_file, "r") as file:
            config = json.load(file)

        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {config['table']}")
        data = cursor.fetchall()
        conn.close()
        logging.info("Data loaded from PostgreSQL successfully.")
        return data
    except FileNotFoundError:
        logging.error("PostgreSQL configuration file not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading data from PostgreSQL: {str(e)}")
        return None


# Configure logging
logging.basicConfig(filename="data_loader.log", level=logging.INFO)

# Example usage
data_csv = load_csv("data.csv")
data_pickle = load_pickle("data.pickle")
data_xls = load_xls("data.xls")
data_postgres = load_postgres("postgres_config.json")
