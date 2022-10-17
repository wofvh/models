from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston
import numpy as np
import time
from pathlib import Path
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target


print(x.shape,y.shape) #(506, 13) (506,)

x = x.reshape(506,13,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )


print(x_train.shape, x_test.shape)  

# (506, 13) (506,)
# (354, 13, 1) (152, 13, 1)


#2. 모델구성
model = Sequential()                
model.add(LSTM(units=100 ,return_sequences=True,activation='relu', input_shape =(13,1)))
model.add(LSTM(units=100 ,return_sequences=False,activation='relu', input_shape =(13,1)))    
model.add(Dense(25, activation='relu'))    
model.add(Dense(15, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(45, activation='relu'))    
model.add(Dense(50, activation='relu'))    
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

# model = load_model(load_filepath + '0707_1753_0096-20.8518.hdf5')


earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 callbacks=[earlyStopping],
                 verbose=1)
#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict.shape)
print(y_test.shape)
# y_predict = y_predict.reshape(102,13)
# y_test = y_test.reshape(102,)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)



# loss :  11.383050918579102
# r2스코어 :  0.8638112999770811

# loss :  23.653762817382812
# r2스코어 :  0.7170024994944979

# LSTM
# loss :  10.43089485168457
# r2스코어 :  0.8737441094302225