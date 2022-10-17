import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM,GRU
from tensorflow.python.keras.layers import Dense, LSTM ,Conv1D ,Flatten
import pandas as pd
#1.데이터 

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7,],[6,7,8,],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

y_predict = np.array([50,60,70])

print(x.shape, y.shape) #x(13, 3) y(13,)

x = x.reshape(13,3,1)


#2. 모델구성                         #cnn = filters  #dnn = units rnn = SimpleRNN 
model = Sequential()                #[batch, timesteps, feature]
# model.add(LSTM(10,input_shape = (3, 1),return_sequences=False))  
model.add(Conv1D(10, input_shape = (3,1)))
model.add(Flatten())
model.add(Dense(3, activation='relu'))   
model.add(Dense(1))
model.summary()   #LSRM