import numpy as np
import pandas as pd
from sqlalchemy import true
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#1. 데이터

path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)    #iddex_col 0 번째 위치함
 
# train_set.info()
# test_set.info()
# print(train_set.shape) #(1459, 10)
train_set.isnull().sum()
test_set.isnull().sum()