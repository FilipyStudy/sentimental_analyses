import pandas as pd
import matplotlib as plt
import sklearn
import re
from pathlib import Path
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import gc
import spacy

#Not random stuff
DB_PATH = Path('database')
REGEX_NUMBER = re.compile('[0-3]')
REGEX_PUNCTUATION = re.compile('[A-Za-z ]')
stopwords = nltk.corpus.stopwords.words('english')

#Regex function to get numbers
def regex_number(x):
    value = REGEX_NUMBER.findall(x)
    return str(value[0])

#Regex function to get out the punctuation
def regex_punctuation(x):
    value = REGEX_PUNCTUATION.findall(x)
    return "".join([i for i in value]).lower()
    

#Define the lambda function to remove stop words from the text
remove_stopwords = lambda x: [word for word in x if word not in stopwords]

#Extract, Transform and Load the data. Aka ETL.
try:
    if DB_PATH.exists():
        train_file = open(r'database/train_data.txt', 'r')

        train_dataframe = pd.read_fwf(train_file, sep=',', header=None, encoding='utf-8')
        
        #Close the files and clean the memory to prevent OOM
        train_file.close()
        gc.collect()

    #Remove NaN column
    train_dataframe = train_dataframe.drop(2, axis=1)
    
    #Rename the header of the DataFrame.
    headers = ['rating', 'review']
    train_dataframe.rename({0: headers[0], 1: headers[1]}, axis=1, inplace=True)

    print('Applying func: regex_number and regex_punctuation')
    #Clean the DataFrame.
    train_dataframe['rating'] = train_dataframe['rating'].apply(regex_number)
    train_dataframe['review'] = train_dataframe['review'].apply(regex_punctuation)

    
    #Tokenize the text.
    print('Applying func: word_tokenize from NLTK')
    train_dataframe['review'] = train_dataframe['review'].apply(word_tokenize)
    
    #Remove StopWords.
    print('Applying func: remove_stopwords')
    train_dataframe['review'] = train_dataframe['review'].apply(remove_stopwords)

    #Save the dataframe as a CSV file for further use.
    train_dataframe.to_csv('train_dataframe.csv', index=False)
except Exception as e:
    print('An exception as ocurred: ' + str(e))
