from re import X
import numpy as np
import pandas as pd
from sqlalchemy import true
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#1. 데이터
# train_set = pd.read_csv('./_data/shopping/train.csv')
# test_set = pd.read_csv('./_data/shopping/test.csv')
# sample_submission = pd.read_csv('./_data/shopping/sample_submission.csv')

path = './_data/shopping/'
test_set = pd.read_csv(path + 'test.csv')    #iddex_col 0 번째 위치함
train_set = pd.read_csv(path + 'test.csv')    #iddex_col 0 번째 위치함
sample_submission = pd.read_csv('./_data/shopping/sample_submission.csv')
# print(train)  #[6255 rows x 13 columns]
# print(test)  #[180 rows x 12 columns]

# print(test_set)           #[6255 rows x 13 columns]
# print(test_set.shape)     #(180, 11)


print(train_set.isnull().sum())
print(test_set.isnull().sum())

train_set = train_set.fillna(0)

print(train_set.isnull().sum())
print(test_set.isnull().sum())

## 결측치 처리1. 제거 ###
def get_month(date):
    month = date[3:5]
    month = int(month)
    return month

train_set['Month'] = train_set['Date'].apply(get_month)
print(train_set)

def holiday_to_number(isholiday):
    if isholiday == True:
        number = 1
    else:
        number = 0
    return number 
train_set['NumberHoliday'] = train_set['IsHoliday'].apply(holiday_to_number)
print(train_set)

test_set = test_set.fillna(0)

test_set['Month'] = test_set['Date'].apply(get_month)
test_set['NumderHolday'] = test_set['IsHoliday'].apply(holiday_to_number)

#2. 모델구성
model = LinearRegression(13)

#분석할 의미가 없는 컬럼은 제거#
train_set = train_set.drop(columns=['id'])
test_set = test_set.drop(columns=['Date','id'])

#전처리 하기 전 컬럼 제거#
train_set = train_set.drop(columns=['Date','IsHoliday'])
test_set = test_set.drop(columns=['Date','IsHoliday'])

#예측하고자 하는 정보 분리
x_train = train_set.drop(columns=['Weekly_Sales'])
y_train = train_set[['Weekly_Sales']]


x = train_set.drop(['Weekly_Sales'])
y = train_set['Weekly_Sales']

print(x.shape, y.shape)


'''
#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping,],
                 verbose=1)

#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
'''