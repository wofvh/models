from unittest import result
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#. 데이터
x = np.array([range(10), range(21,33), range(201, 211)])
print((range(10))) #range 범위, 거리 8-10까지의 정수형 숫자

for i in range(10): # for :반복하라
    print(i) 
 
print(x.shape) # (3, 10)
x = np.transpose(x)
print(x.shape) # (10, 3)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])

y = np.transpose(y)
print(x.shape)    #(10, 2)
