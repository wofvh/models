import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.layers import Bidirectional
import pandas as p
from tensorflow.keras.layers import Dense, SimpleRNN ,LSTM,GRU

#1.데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7,],[6,7,8,],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

y_predict = np.array([50,60,70])

print(x.shape, y.shape) #x(13, 3) y(13,)

x = x.reshape(13,3,1)
#모델구성
model = Sequential()                #[batch, timesteps, feature]
model.add(Bidirectional(SimpleRNN(64, return_sequences=True), input_shape=(3, 1)))
model.add(LSTM(15, return_sequences=True))  
model.add(Bidirectional(GRU(10)))
model.add(Dense(36, activation='relu'))    
model.add(Dense(33, activation='relu'))    
model.add(Dense(37, activation='relu'))    
model.add(Dense(39, activation='relu'))    
model.add(Dense(38, activation='relu'))    
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss ='mse',optimizer='adam')
model.fit(x,y, epochs=800)

# from tensorflow.python.keras.callbacks import EarlyStopping 
# earlystopping = EarlyStopping(monitor='loss',patience=20, mode='min', verbose=1,
#               restore_best_weights=True)

#4. 평가 예측
loss = model.evaluate(x,y)
y_predict = np.array([50,60,70]).reshape(1, 3,1) #[[[8],[9],[10]]]
result = model.predict(y_predict)  #평가예측에서 똑같이 맟춰서
print('loss:',loss)
print('[50,60,70]의 결과:',result)  #RNN input_shape 에서 들어간 차원을 야함