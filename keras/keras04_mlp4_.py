#mlp = 멀티 레이어 퍼셉트론
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#. 데이터
x = np.array([range(1,1039)])
print(x.shape)




for i in range(1039): 
     print(i) 

x = np.transpose(x)
print(x.shape) # (10, 1)


# y = np.array([[1,2,3,4,5,6,7,8,9,10],
#              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
#              [9,8,7,6,5,4,3,2,1,0]])
# y = np.transpose(y)
# print(y.shape) 

# model = Sequential()
# model.add(Dense(50, input_dim=1))
# model.add(Dense(89))
# model.add(Dense(104))
# model.add(Dense(122))
# model.add(Dense(123))
# model.add(Dense(3, ))

# # 컴파일 훈련
# model.compile(loss='mae', optimizer='adam')
# model.fit(x, y, epochs=500, batch_size=1) 

# #. 평가,예측 
# loss = model.evaluate(x, y)
# print('loss : ',loss)

# reuslt = model.predict([9])
# print('[9]의 예측값 : ', reuslt) 

#2 모델
#[실습]

# 예측 : [[9]]  -> 예상 y값 [[10, 1.9, 0]]

