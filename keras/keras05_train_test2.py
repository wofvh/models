import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#과제 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라.
x_train = x [:7]
x_test = x [7:]
y_train = x [:7]
y_test = x [7:]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#나와야 하는 결과치
#x_train = np.array([1,2,3,4,5,6,7)
#x_test = np.array([8,9,10])
#y_train = np.array([1,2,3,4,5,6,7])
#y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1) 

#평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)
result = model.predict([11])
print('11의 예측값 : ', result)


# loss :  0.0012542374897748232
# 11의 예측값 :  [[11.01016]]