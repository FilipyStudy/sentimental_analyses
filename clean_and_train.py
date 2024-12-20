import pandas as pd
import matplotlib as plt
import sklearn
import re
from pathlib import Path
import random
import numpy as np

DB_PATH = Path('database')
REGEX_NUMBER = re.compile('[0-3]')
REGEX_PUNCTUATION = re.compile('[A-Za-z ]')


def regex_number(x):
    value = REGEX_NUMBER.findall(x)
    return str(value[0])


def regex_punctuation(x):
    value = REGEX_PUNCTUATION.findall(x)
    return "".join([i for i in value])
    

#Extract, Transform and Load the data. Aka ETL.
try:
    if DB_PATH.exists():
        train_file = open(r'database/train_data.txt', 'r')
        test_file = open(r'database/test_data.txt', 'r')

        train_dataframe = pd.read_fwf(train_file, sep=',', header=None, encoding='utf-8')
        test_dataframe = pd.read_fwf(test_file, sep=',', header=None, encoding='utf-8')
        
        #Remove NaN column
        train_dataframe = train_dataframe.drop(2, axis=1)

#Rename the header of the DataFrame.
    headers = ['rating', 'review']
    train_dataframe.rename({0: headers[0], 1: headers[1]}, axis=1, inplace=True)
    test_dataframe.rename({0: headers[0], 1: headers[1]}, axis=1, inplace=True)

    #Clean the DataFrame.
    train_dataframe['rating'] = train_dataframe['rating'].apply(regex_number)
    test_dataframe['rating'] = test_dataframe['rating'].apply(regex_number)
    train_dataframe['review'] = train_dataframe['review'].apply(regex_punctuation)
    test_dataframe['review'] = test_dataframe['review'].apply(regex_punctuation)

    #Print the DataFrame for debugging.
    print(train_dataframe.head())
    print(test_dataframe.head())

except Exception as e:
    print('An exception as ocurred: ' + str(e))
