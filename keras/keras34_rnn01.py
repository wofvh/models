import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
import pandas as pd

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# x, y = zip(['1,2,3', 4], ['4,5,6', 7], ['7,8,9', 10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)  # x(7 , 3) y(7,)

# input_shape = (행 , 열 , 몇개씩 자르는지! Rnn 쉐이프 3 차원 )
x = x.reshape
print(x.shape)#(7,3,1,)


#2. 모델구성 
model = Sequential()  
model.add(SimpleRNN(64,input_shape =(3,1)))  #삼차원으로 들어감)
# model.add(SimpleRNN(32))
model.add(Dense(16, activation='relu'))    #(RNN 여기서2차원으로 바뀜)
model.add(Dense(42, activation='relu'))    #(RNN 여기서2차원으로 바뀜)
model.add(Dense(55,activation='relu') )   #(RNN 여기서2차원으로 바뀜)
model.add(Dense(38, activation='relu'))    #(RNN 여기서2차원으로 바뀜)
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss ='mse',optimizer='adam')
model.fit(x,y, epochs=1000)

#4. 평가 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1) #[[[8],[9],[10]]]
result = model.predict(y_pred)
print('loss:',loss)
print('[8,9,10]의 결과:',result)  #RNN input_shape 에서 들어간 차원을 야함  
                                  #평가예측에서 똑같이 맟춰서
                                  