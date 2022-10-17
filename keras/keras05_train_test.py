from audioop import add
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from re import X
from unittest import result
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1.데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=1))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1) 

#평가, 예측
loss = model.evaluate(x_test, y_test) #X.Y
print('loss : ',loss)
results = model.predict([11])
print('11의 예측값 : ', results)


# 11의 예측값 :  [[11.000111]]