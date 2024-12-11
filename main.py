import pandas as pd
import matplotlib as plt
import sklearn
import re
from pathlib import Path
import random
import numpy as np


DB_PATH = Path('database')
REGEX_NUMBER = re.compile('[0-3]')

#Extract, Transform and Load the data. Aka ETL.
try:
    if DB_PATH.exists():
        train_file = open(r'database/train_data.txt', 'r')
        test_file = open(r'database/test_data.txt', 'r')

        train_dataframe = pd.read_fwf(train_file, sep=',', header=None, encoding='utf-8')
        test_dataframe = pd.read_fwf(test_file, sep=',', header=None, encoding='utf-8')
        
        train_dataframe.drop([2])

    def regex_number(x):
        value = REGEX_NUMBER.findall(x)
        return str(value[0])


    train_dataframe[0].apply(regex_number)
    test_dataframe[0].apply(regex_number)
    print(train_dataframe.head())
    print(test_dataframe.head())

except Exception as e:
    print('An exception as ocurred: ' + str(e))
