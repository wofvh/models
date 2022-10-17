from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
import numpy as np
import time
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM


#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target

print(x.shape) # x(150, 4) (120, 4) (30, 4)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

print(x.shape) # x(150, 4) (120, 4) (30, 4)
print(y.shape)

x = x.reshape(150,4,1)
print(x_train.shape, x_test.shape) 
print(np.unique(y, return_counts=True))
'''
#2. 모델구성

model = Sequential()
model.add(LSTM(units=100 ,return_sequences=True,
               activation='relu', input_shape =(4,1)))
model.add(LSTM(units=100 ,return_sequences=False,
               activation='relu',))
model.add(Dense(25, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(45, activation='relu'))    
model.add(Dense(50, activation='relu'))    
model.add(Dense(3))
model.summary()


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict.shape)
print(y_test.shape)

y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

print(y_test.shape)
print(y_predict.shape)
print(y_test)
print(y_predict)

from sklearn.metrics import accuracy_score, r2_score

acc= accuracy_score(y_test, y_predict)
print('loss : ' , loss)
print('acc스코어 : ', acc) 

# loss :  [0.014415979385375977, 1.0]
# acc스코어 :  1.0
'''