from colorsys import yiq_to_rgb
import xdrlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import MaxPooling1D, GRU, Activation, Dense, Conv1D, Reshape, LSTM, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Bidirectional
import time
start = time.time()


#1. 데이터
path = './_data/kaggle_jena/'
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv') # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)

# datasets['Date Time'] = datasets['Date Time'].astype('str')
# datasets['Date Time'].dtypes
# date_list = datasets['Date Time'].str.split('-')
# date_list.head()

# datasets['month']
# df = pd.DataFrame(datasets)
datasets = datasets.drop(['Date Time'],axis=1)
datasets = np.transpose(datasets)
df_test = np.array(datasets)
df_test = df_test[:,1:]
print(datasets)
print(df_test)
print(df_test.shape) #(14, 420550)

print(datasets.shape) # (14, 420551)
# print(train_set.describe())
# print(train_set.columns)
df = pd.DataFrame(datasets)
print(df.columns)

# x = df.drop(['420551'], axis=1)
# y = df['420551']
df = np.array(df)
x = df[:,:-1]
y = df[:,-1]
print(x)
print(y)


print(x.shape) #(14, 420550)
print(y.shape) #(14,)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)
df_test = scaler.transform(df_test)

###################리세이프#######################
x = x.reshape(14, 647, 650)
df_test = df_test.reshape(14, 647, 650)
print(x.shape)
# print(np.unique(y_train, return_counts=True))
#################################################


#2. 모델구성
model = Sequential()
model.add(Conv1D(120,3, input_shape=(647,650)))
model.add(MaxPooling1D())
model.add(Conv1D(120,3))
model.add(MaxPooling1D())
model.add(Bidirectional(GRU(160, return_sequences=True)))
model.add(Bidirectional(GRU(160, return_sequences=True)))
model.add(Bidirectional(LSTM(60)))
model.add(Dense(180, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(Dense(1))
model.summary()  



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

# save_filepath = './_ModelCheckPoint/' + current_name + '/'
# load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience= 1000, mode='auto', verbose=1, 
                              restore_best_weights=True)        

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath= "".join([save_filepath, date, '_', filename])
#                       )

hist = model.fit(x, y, epochs=2000, batch_size=300,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y) 
y_predict = model.predict(x)
y_summit = model.predict(df_test)

print(y_predict.shape) #(14, 16, 1)
print(y_summit.shape) # (14, 16, 1)
print(df_test.shape) # (14, 647, 650)


# acc = accuracy_score(y, y_predict)
print('loss : ', loss)
print('2017.01.01 00:10:00의 날씨 : ', y_summit)
# print('acc스코어 : ', acc)
print("time :", time.time() - start)

