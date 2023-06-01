"""
preprocessing.py

This module contains functions for preprocessing textual data, including text cleaning, 
spelling correction, and data preprocessing. Please note that the code and data in this 
repository have been modified for confidentiality purposes, preserving the sensitive 
details of the project that generated the article. As a result, some functions may appear 
more generic, but they are fully functional and demonstrate all the concepts discussed in 
the article.

Functions:
- clean_text(text): Cleans the given text by removing special characters, digits, and 
  stopwords, and performs stemming.
- correct_text(text): Corrects spelling and grammar errors in the given text.
- tokenize_texts(texts, tokenizer): Tokenizes a list of texts using a tokenizer.
- encode_tokens(tokenized_texts, tokenizer, max_length): Encodes tokenized texts into 
  numerical IDs using a tokenizer.
- get_semantic_embeddings(input_ids, model): Calculates semantic embeddings from input 
  IDs using a pre-trained model.
- save_embeddings(embeddings, output_file): Saves semantic embeddings to a CSV file.
- preprocess_data(input_file, output_file, model_name='bert-base-uncased', max_length=128): 
  Preprocesses text data,generates semantic embeddings using a pre-trained BERT model, and 
  saves the embeddings to a CSV file.

Example Usage:
data_csv = load_csv('data.csv')
data_pickle = load_pickle('data.pickle')
data_xls = load_xls('data.xls')
data_postgres = load_postgres('postgres_config.json')

preprocessed_data = preprocess_data(data_csv)
print(preprocessed_data.head())

Authors: This code was developed by the Data Science Laboratory at the Polytechnic School 
of the University of São Paulo, led by researchers Wellingthon Queiroz and Osvaldo Gogliano, 
as described in the article.
         Main contacts:
            - Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
            - Osvaldo Gogliano - ogogli@gmail.com
version: 0.15rc

"""



import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import logging
from data_loader import load_csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from autocorrect import Speller

logging.basicConfig(level=logging.INFO)

def clean_text(text):
    """
    Cleans the given text by removing special characters, digits, stopwords, and performs stemming.

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Cleaned text.

    """
  
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
  
    # Convert to lowercase
    text = text.lower()
  
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
  
    # Stemming
    stemmer = SnowballStemmer('english')
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
  
    # Join tokens back into text
    cleaned_text = ' '.join(stemmed_tokens)
  
    return cleaned_text

def correct_text(text):
    """
    Corrects spelling and grammar errors in the given text.

    Args:
        text (str): Text to be corrected.

    Returns:
        str: Corrected text.

    """
    # Split text into words
    words = text.split()
  
    # Initialize the English dictionary
     # The article was written in Brazil and applied to a Portuguese text database. 
     # Remember to change it to your language.
    dictionary = enchant.Dict("pt_BR")
    # Initialize the spell checker
    spell = Speller(lang='pt-br')
  
    # Check and correct each word
    corrected_words = [spell(word) if not dictionary.check(word) else word for word in words]
  
    # Join corrected words back into text
    corrected_text = ' '.join(corrected_words)
  
    return corrected_text

def tokenize_texts(texts, tokenizer):
    """
    Tokenizes a list of texts using a tokenizer.

    Args:
        texts (list): List of texts to be tokenized.
        tokenizer (object): Tokenizer object to be used for tokenization.

    Returns:
        list: List of tokenized texts.

    """
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    
    return tokenized_texts

def encode_tokens(tokenized_texts, tokenizer, max_length):
    """
    Encodes tokenized texts into numerical IDs using a tokenizer.

    Args:
        tokenized_texts (list): List of tokenized texts.
        tokenizer (object): Tokenizer object to be used for encoding.
        max_length (int): Maximum length of the input sequence.

    Returns:
        torch.Tensor: Padded input IDs with a shape of (batch_size, max_length).

    """
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]
    padded_input_ids = torch.tensor([ids + [0]*(max_length - len(ids)) for ids in input_ids])
    
    return padded_input_ids

def get_semantic_embeddings(input_ids, model):
    """
    Calculates semantic embeddings from input IDs using a pre-trained model.

    Args:
        input_ids (torch.Tensor): Padded input IDs.
        model (object): Pre-trained model object to be used for embedding extraction.

    Returns:
        numpy.ndarray: Array of semantic embeddings.

    """
    with torch.no_grad():
        outputs = model(input_ids)
    
    semantic_embeddings = outputs[0][:, 0, :].numpy()
    
    return semantic_embeddings

def save_embeddings(embeddings, output_file):
    """
    Saves semantic embeddings to a CSV file.

    Args:
        embeddings (numpy.ndarray): Array of semantic embeddings.
        output_file (str): File path for saving the embeddings.

    """
    semantic_data = pd.DataFrame(embeddings)
    semantic_data.to_csv(output_file, index=False)
    logging.info(f"Semantic embeddings saved to {output_file}.")

def preprocess_data(input_file, output_file, model_name='bert-base-uncased', max_length=128):
    """
    Preprocesses text data, generates semantic embeddings using a pre-trained BERT model,
    and saves the embeddings to a CSV file.

    Args:
        input_file (str): File path of the input dataset.
        output_file (str): File path for saving the semantic embeddings.
        model_name (str, optional): Name of the pre-trained BERT model. Defaults to 'bert-base-uncased'.
        max_length (int, optional): Maximum length of the input sequence. Defaults to 128.

    """
    # Load the dataset
    data = load_csv(input_file)
    if data is None:
        return

    # Preprocess the text data
    data['clean_description'] = data['description'].apply(clean_text)
    data['clean_description'] = data['clean_description'].apply(correct_text)

    # Load the pre-trained BERT model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize the description text
    tokenized_texts = tokenize_texts(data['clean_description'], tokenizer)

    # Encode the tokens into numerical IDs
    padded_input_ids = encode_tokens(tokenized_texts, tokenizer, max_length)

    # Get the semantic embeddings
    semantic_embeddings = get_semantic_embeddings(padded_input_ids, model)

    # Save the semantic embeddings
    save_embeddings(semantic_embeddings, output_file)


# Example usage
preprocess_data('seu_dataset.csv', 'semantic_embeddings.csv')



# Example usage
data_csv = load_csv('data.csv')
data_pickle = load_pickle('data.pickle')
data_xls = load_xls('data.xls')
data_postgres = load_postgres('postgres_config.json')

preprocessed_data = preprocess_data(data_csv)
print(preprocessed_data.head())