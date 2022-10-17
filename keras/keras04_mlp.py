from unittest import result
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]
             )
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) # (2, 10)
print(y.shape) # (10, )

x = x.T
# x = x.transpose()
# x = x.T.reshape(10,2)
print(x)
print(x.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(23))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=450, batch_size=1) 

#. 평가,예측 
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[10,1.4]])
print('[10, 1.4]의 예측값 : ', result)

#[10, 1.4]의 예측값 :  [[20.000002]]

