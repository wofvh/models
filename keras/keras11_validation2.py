from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np 

x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[1:15]
y_train = y[1:15]

x_test = x[5:12]
y_test = y[5:12]

x_val = x[1:11]
y_val = y[1:11]

print(x_train)
print(x_test)
print(x_val)


# x_train = np.array(range(1, 11))
# y_train = np.array(range(1, 11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])

model = Sequential()
model.add(Dense(128, input_dim = 1))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#컴파일 , 훈련 
model.compile(loss ='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=5,
          validation_data=(x_val,y_val))

#4. 평가.예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print("17의 예측값",result)


# loss: 6.158425094326958e-05
# 17의 예측값 [[16.98115]]