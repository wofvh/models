from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
#데이터
x = np.array(range(1,17)) #array  변수에 여러가지 값을 저장때
y = np.array(range(1,17)) #range 범위
x_train, x_test_val, y_train, y_test_val = train_test_split(x,y,
                                                 test_size=0.2,
                                                 random_state=66
                                                 )

x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val,
                                                 train_size=0.5,
                                                 random_state=66
                                                 )

print(y_train)
print(y_test)
print(y_val)
print(x_train)
print(x_test)
print(x_val)

# [10  3 15  6 12 11  4  8 13  5]
# [16  1  9]
# [ 7  2 14]                                 



print(x_train) #[2 7 6 3 4 8 5]

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
          validation_split=0.25) #validation_
#4. 평가.예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print("17의 예측값",result)
# loss: 0.00019022470223717391
# 17의 예측값 [[16.980562]]