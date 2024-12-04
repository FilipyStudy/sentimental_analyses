import pandas as pd
import matplotlib as plt
import sklearn
import re
from pathlib import Path
import random
import numpy as np

q = Path('database')
    
try:
    if q.exists():
        train_file = open(r'database/train_data.txt', 'r')
        test_file = open(r'database/test_data.txt', 'r')

        train_dataframe = pd.read_csv(train_file, sep='', header=None)
        test_dataframe = pd.read_csv(test_file, sep='', header=None)

except Exception as e:
    print('An exception as ocurred: ' + str(e))

finally:
    print("Script dosn't run")
