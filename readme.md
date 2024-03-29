# Research Article

This repository is developed to support the research article titled **"INTELIGÊNCIA ARTIFICIAL E PROCESSAMENTO DE LINGUAGEM NATURAL NO MONITORAMENTO DA ESTRADA DE FERRO VITÓRIA – MINAS"**. 
The article investigates the application of Natural Language Processing (NLP) techniques to identify occurrences in multiple datasets, including those manually filled with various errors such as typographical, semantic, and spelling mistakes.

The main objective of this research is to find occurrences in these datasets and then cross-reference the occurrence data with the instrumented wagon data to gain insights into the behavior of the sensors during the occurrences. 
The project utilizes NLP techniques for text preprocessing, occurrence identification, and semantic analysis. Various machine learning models, such as logistic regression and support vector machines, are employed for classification tasks. 
The evaluation includes performance metrics such as accuracy, precision, recall, and F1-score. Additionally, techniques like k-fold cross-validation and confusion matrix analysis are used for model validation and performance assessment.

The research aims to contribute to the field of railway monitoring by leveraging AI and NLP techniques to enhance the analysis of instrumented wagon data and improve the identification and understanding of occurrences. By combining data from different sources and applying advanced NLP algorithms, this project provides valuable insights into the occurrences and their impact on sensor readings, facilitating efficient maintenance and decision-making processes.


## Pipeline
The pipeline consists of the following steps:

1. Data loading from various sources: CSV files, Pickle files, XLS files, and PostgreSQL databases.
2. Text preprocessing: cleaning, spelling correction, and data preprocessing using NLP techniques.
3. Model training and evaluation: employing machine learning algorithms and NLP-based classifiers to classify occurrences.
4. False positive detection: identifying false positives in the classification results.
5. Data visualization and analysis: exploring and analyzing the dataset using NLP-driven techniques.

## Project Structure

The project is structured into the following files and directories:

- `data_loader.py`: Contains functions for loading data from various sources.
- `preprocessing.py`: Contains functions for text preprocessing using NLP techniques.
- `train_test.py`: Contains functions for splitting the data into train and test sets using NLP-based strategies.
- `classifier.py`: Contains functions for training and evaluating a classifier model using NLP features.
- `detect_false_positive.py`: Contains functions for detecting false positives in the classification results.
- `utils.py`: Contains utility functions used throughout the project.
- `notebook.ipynb`: Jupyter Notebook file with complete code and experiments.
- `data/`: Directory to store  the dataset files.
- `models/`: Directory to store trained models.
- `results/`: Directory to store experiment results.

### IMPORTANT:

Please note that the code and data in this repository have been modified for confidentiality purposes, preserving the sensitive details of the project that generated the article. As a result, some functions may appear more generic, but they are fully functional and demonstrate all the concepts discussed in the article.


- Authors: 
This code was developed by the Data Science Laboratory at the Polytechnic School 
of the University of São Paulo, led by researchers Wellingthon Queiroz and Osvaldo Gogliano, 
as described in the article.
         Main contacts:
            - Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
            - Osvaldo Gogliano - ogogli@gmail.com
- version: 0.15rc

## Usage

To run the project, follow these steps:

1. Install the required dependencies listed in `requirements.txt`.
2. Place the dataset files in the `data/` directory.
3. Use the code as needed.


Make sure to update the file paths and configurations in the code as needed.

#
## Authors

This code was developed by the **Data Science Laboratory at the Polytechnic School of the University of São Paulo**, led by researchers Wellingthon Queiroz and Osvaldo Gogliano.

- Wellingthon Queiroz (Tony Dias) - tonydiasq@gmail.com
- Osvaldo Gogliano - ogogli@gmail.com

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
