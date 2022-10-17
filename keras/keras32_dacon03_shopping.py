from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd   # 엑셀 데이터 불러올 때 사용
from pandas import DataFrame 
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.utils import to_categorical 
from keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation


# train 데이터 불러오기
train_set = pd.read_csv('./_data/shopping/train.csv')

# test 데이터 불러오기
test_set = pd.read_csv('./_data/shopping/test.csv')

# sample_submission 불러오기
submission_set = pd.read_csv('./_data/shopping/sample_submission.csv')

# plt.hist(train.Weekly_Sales, bins=50)
# plt.show()

train_set = train_set.fillna(0)

def get_day(date):
    day = date[0:2]
    day = int(day)
    return day
train_set['Day'] = train_set['Date'].apply(get_day)

def get_month(date):
    month = date[3:5]
    month = int(month)
    return month
train_set['Month'] = train_set['Date'].apply(get_month)

def get_year(date):
    year = date[7:3]
    year = int(year)
    return year
train_set['Year'] = train_set['Date'].apply(get_year)

# 결측치 처리
test_set = test_set.fillna(0)

# Date 전처리
test_set['Day'] = test_set['Date'].apply(get_day)
test_set['Month'] = test_set['Date'].apply(get_month)
test_set['Year'] = test_set['Date'].apply(get_year)

# 분석할 의미가 없는 칼럼을 제거합니다.
train_set = train_set.drop(columns=['id'])
test_set = test_set.drop(columns=['id'])

# 전처리 하기 전 칼럼들을 제거합니다.
train_set = train_set.drop(columns=['Date','IsHoliday'])
test_set = test_set.drop(columns=['Date','IsHoliday'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
# x_train = train_set.drop(columns=['Weekly_Sales'])
# y_train = train_set[['Weekly_Sales']]

x = train_set.drop(columns=['Weekly_Sales']) # .drop - 데이터에서 ''사이 값 빼기, # axis=1 (열을 날리겠다), axis=0 (행을 날리겠다)
print(x)
print(x.columns)
print(x.shape)  # 

y = train_set['Weekly_Sales']
print(y)
print(y.shape)  # 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=31)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성 
# model = Sequential()
# model.add(Dense(64, input_dim=12))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(16))
# model.add(Dense(1))
input1 = Input(shape=(12,))
dense1 = Dense(200)(input1)
batchnorm1 = BatchNormalization()(dense1)
activ1 = Activation('relu')(batchnorm1)
drp4 = Dropout(0.2)(activ1)
dense2 = Dense(100)(drp4)
batchnorm2 = BatchNormalization()(dense2)
activ2 = Activation('relu')(batchnorm2)
drp5 = Dropout(0.2)(activ2)
dense3 = Dense(100)(drp5)
batchnorm3 = BatchNormalization()(dense3)
activ3 = Activation('relu')(batchnorm3)
drp6 = Dropout(0.2)(activ3)
output1 = Dense(1)(drp6)
model = Model(inputs=input1, outputs=output1)   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.15, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict): #괄호 안의 변수를 받아들인다 :다음부터 적용 
    return np.sqrt(mean_squared_error(y_test, y_predict)) #루트를 씌워서 돌려줌 

rmse = RMSE(y_test, y_predict)  #y_test와 y_predict를 비교해서 rmse로 출력 (원래 데이터와 예측 데이터를 비교) 
print("RMSE : ", rmse)
