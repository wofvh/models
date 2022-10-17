import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM,GRU
import pandas as pd
#1.데이터 

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7,],[6,7,8,],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(y.shape , x.shape)   #y(13,) x(13, 3)

#2. 모델구성

model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape = (3,1)))
model.add(LSTM(5))
model.add(Dense(1))
model.summary()    # (n,3,1) > (n,3,10) # 


# _________________________________________________________________     
# Layer (type)                 Output Shape              Param #   
# =================================================================     
# lstm (LSTM)                  (None, 3, 10)             480
# _________________________________________________________________     
# lstm_1 (LSTM)                (None, 5)                 320
# _________________________________________________________________     
# dense (Dense)                (None, 1)                 6
# =================================================================     
# Total params: 806
# Trainable params: 806
# Non-trainable params: 0
# _________________________________________________________________   