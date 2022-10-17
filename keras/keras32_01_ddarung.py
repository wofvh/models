import numpy as np
import pandas as pd
from sqlalchemy import true
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#1. 데이터

path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
print(train_set)
print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv',index_col=0)    #iddex_col 0 번째 위치함
print(test_set)
print(test_set.shape)  # (715, 9)

print(train_set.columns)
print(train_set.info())      #unll 중간중간 없는데이터 #결측치 이빨빠진 데이터
print(train_set.describe()) 

### 결측치 처리1. 제거 ###
print(train_set.isnull().sum())
train_set = train_set.fillna(method = 'ffill')
test_set = test_set.fillna(method = 'ffill')
print(train_set.isnull().sum())

x = train_set.drop(['count'], axis=1)     #drop 지울떄 사용함
print(x)
print(x.columns)
print(x.shape)       #(1459, 9)

y = train_set['count']
print(y)
print(y.shape)       #(1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=777 )


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(100, activation ='selu'))
model.add(Dense(100, activation ='selu'))
model.add(Dense(100, activation ='selu'))
model.add(Dense(100, activation ='selu'))
model.add(Dense(1))


#3. 컴파일 , 훈련 
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =500, batch_size=100 )

#평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape) #(715, 1)

submission_set = pd.read_csv(path + 'submission.csv', index_col=0)

submission_set['count'] = y_summit
print(submission_set)

submission_set.to_csv('./_data/ddarung/submission.csv',index= True)


#.to_csv()를 사용해서
## submission을 완성하기 




#y_predict = model.predict(test_set)

#함수 정의 하기

# loss :  1178.3143310546875
# RMSE :  34.32658399508923
