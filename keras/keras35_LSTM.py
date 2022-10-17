import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM
import pandas as pd
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Sequential,load_model,Model

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# x, y = zip(['1,2,3', 4], ['4,5,6', 7], ['7,8,9', 10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)  # x(7 , 3) y(7,)

# input_shape = (행 , 열 , 몇개씩 자르는지! Rnn 쉐이프 3 차원 )
x = x.reshape(7,3,1)
# print(x.shape)#(7,3,1,)


#2. 모델구성                         
# model.add(SimpleRNN(units=10 ,input_length=3 ,input_dim=1))
# model.add(SimpleRNN(units=10 ,input_dim=1 ,input_length=3)) #가동성이 떨어짐 
# model.add(LSTM(units=20 ,input_length=3 ,input_dim=1))
model = Sequential()                
model.add(LSTM(units=10 ,input_shape =(3,1)))  
model.add(Dense(25, activation='relu'))    
model.add(Dense(15, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(45, activation='relu'))    
model.add(Dense(50, activation='relu'))    
model.add(Dense(1))
model.summary()

# _________________________________________________________________     
# Layer (type)                 Output Shape              Param #        
# =================================================================     
# lstm (LSTM)                  (None, 10)                480
# _________________________________________________________________     
# dense (Dense)                (None, 25)                275
# _________________________________________________________________     
# dense_1 (Dense)              (None, 15)                390
# _________________________________________________________________     
# dense_2 (Dense)              (None, 35)                560
# _________________________________________________________________     
# dense_3 (Dense)              (None, 45)                1620
# _________________________________________________________________     
# dense_4 (Dense)              (None, 50)                2300
# _________________________________________________________________     
# dense_5 (Dense)              (None, 1)                 51
# =================================================================     
# Total params: 5,676
# Trainable params: 5,676
# Non-trainable params: 0
# _________________________________________________________________  


#SimpleRNN *연산법 units * (feature + bisa + units) = parms 예units이10 일때
#RNN *연산법 units * (feature + bisa + units) = parms 
#LSTM 숫자 4의 의미는 * (cell_state, input_gate, output_gate , forget_gate
#LMST 연산법 4 * 10 * (1 + 1 + 10) = 480 ////// 예) 4 * 20 * (1 + 1 + 20) = 1700





#3.컴파일 훈련
model.compile(loss ='mse',optimizer='adam')
model.fit(x,y, epochs=600)

#4. 평가 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1) #[[[8],[9],[10]]]
result = model.predict(y_pred)  #평가예측에서 똑같이 맟춰서
print('loss:',loss)
print('[8,9,10]의 결과:',result)  #RNN input_shape 에서 들어간 차원을 야함

# loss: 9.897771633404773e-06
# [8,9,10]의 결과: [[10.943413]]