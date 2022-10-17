from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.datasets import fetch_california_housing
import time
start = time.time()
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################



#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

print(x_train.shape,x_test.shape)
x_train = x_train.reshape(16512, 8, 1)
x_test = x_test.reshape(4128, 8, 1)


#2. 모델구성

# model = load_model("./_save/keras22_hamsu02_california.h5")

# model = Sequential()
# model.add(Dense(20, input_dim=8,activation='sigmoid'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(8,1))
conv1d1 = Conv1D(20,2)(input1)
flat = Flatten()(conv1d1)
dense3 = Dense(50, activation='relu')(flat)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)



save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/keras25_MCP_2_california.py/'
# model = load_model(load_filepath + '0707_1753_0096-20.8518.hdf5')

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
print("time :", time.time() - start)

# DNN
# loss :  [0.2925138473510742, 0.33528944849967957]
# r2스코어 :  0.7901720946372207

# Conv1D
# loss :  0.3401532471179962
# r2스코어 :  0.7559990021223169
# time : 379.8870494365692