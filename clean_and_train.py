import pandas as pd
import matplotlib as plt
import sklearn
import re
from pathlib import path
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

#not random stuff
DB_PATH = path('database')
REGEX_NUMBER = re.compile('[0-3]')
REGEX_PUNCTUATION = re.compile('[a-zA-Z ]')
stopwords = nltk.corpus.stopwords.words('english')

#regex function to get numbers
def regex_number(x):
    value = REGEX_NUMBER.findall(x)
    return str(value[0])

#regex function to get out the punctuation
def regex_punctuation(x):
    value = REGEX_PUNCTUATION.findall(x)
    return "".join([i for i in value]).lower()
    

#define the lambda function to remove stop words from the text
remove_stopwords = lambda x: [word for word in x if word not in stopwords]

#extract, transform and load the data. aka etl.
try:
    if DB_PATH.exists():
        train_file = open(r'database/train_data.txt', 'r')
        test_file = open(r'database/test_data.txt', 'r')

        train_dataframe = pd.read_fwf(train_file, sep=',', header=none, encoding='utf-8')
        test_dataframe = pd.read_fwf(test_file, sep=',', header=none, encoding='utf-8')
       
        #close the database to save memory and prevent oom on my os
        train_file.close()
        test_file.close()
        #remove nan column
        train_dataframe = train_dataframe.drop(2, axis=1)

#rename the header of the dataframe.
    headers = ['rating', 'review']
    train_dataframe.rename({0: headers[0], 1: headers[1]}, axis=1, inplace=true)
    test_dataframe.rename({0: headers[0], 1: headers[1]}, axis=1, inplace=true)

    #clean the dataframe.
    train_dataframe['rating'] = train_dataframe['rating'].apply(regex_number)
    test_dataframe['rating'] = test_dataframe['rating'].apply(regex_number)
    train_dataframe['review'] = train_dataframe['review'].apply(regex_punctuation)
    test_dataframe['review'] = test_dataframe['review'].apply(regex_punctuation)

    #tokenize the text
    train_dataframe['review'] = train_dataframe['review'].apply(word_tokenize)
    test_dataframe['review'] = test_dataframe['review'].apply(word_tokenize)
    
    #remove stop words.
    train_dataframe['review'] = train_dataframe['review'].apply(remove_stopwords)
    test_dataframe['review'] = test_dataframe['review'].apply(remove_stopwords)

    #print the dataframe for debugging.
    print(train_dataframe.head())
    print(test_dataframe.head())

except exception as e:
    print('An exception as ocurred: ' + str(e))
