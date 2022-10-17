import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
import numpy as np
x = np. array([1,2,3,5,4]) #array 변수에 여러 개의 값을 저장할때
y = np. array([1,2,3,4,5])

#실습 6

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()   #신경망(node) 1.2.3 (데이터) *너무 많이 작동하면 메모리 부화 
model.add(Dense(5, input_dim=1)) #INPUT을 하나씩 넣어서 메모리를 줄일수있음 (batch 작업) 
model.add(Dense(256)) #batch를 줄일때 장점 메모리를 적게함 *시간이 엄청 많이걸림 *loss가 줄어듬
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mae',optimizer='adam') #(mse 평균제곱오차) (loss 오차)compile (Optimizer(최적화) (mae절대값 )
model.fit(x, y, epochs=64)

loss = model.evaluate(x, y) #평가값을 로스에 넣는다 #(evaliate 평가)
print('loss : ',loss)

result = model.predict([6])
print('6의 예측값 : ',result )

# loss :  0.44082245230674744
# 6의 예측값 :  [[6.0140486]]