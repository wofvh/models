from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np
import time
from pathlib import Path
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM

#1. 데이터
datasets = fetch_california_housing()

x, y = datasets.data, datasets.target


print(x.shape,y.shape) #(20640, 8) (20640,)

x = x.reshape(20640,8,1)

print(x.shape, y.shape) #(20640, 8, 1) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

print(x_train.shape, x_test.shape)  # (14447, 8, 1) (6193, 8, 1)

#2. 모델구성

model = Sequential()                
model.add(LSTM(units=100 ,return_sequences=True,
               activation='relu', input_shape =(8,1)))
model.add(LSTM(units=100 ,return_sequences=False,
               activation='relu'))
model.add(Dense(35, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(30, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=200, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict.shape)
print(y_test.shape)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)


# loss :  11.383050918579102
# r2스코어 :  0.8638112999770811

# loss :  0.37719547748565674
# r2스코어 :  0.7294276537739492