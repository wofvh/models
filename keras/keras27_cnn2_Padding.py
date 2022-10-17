from turtle import setx
from wsgiref.simple_server import demo_app
from numpy import kaiser
from psutil import disk_io_counters
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D

#input with shape (batch_size, input_dim) 이차원일때

model = Sequential()

# model.add(Dense(units=10, input_shape=(3,)))
# model.summary()
#(input_dim + bias) * units = summary param (Dense모델 )
#5X5 이미지(1 흑백 3 컬러 ) (rows, columns,channels)
                                                                
model.add(Conv2D(filters=64, kernel_size=(3, 3), #(4,4,10) 
                 padding='same',
                 input_shape=(28, 28, 1)))  #kernel_size 이미지 분석을 위해 2x2로 잘라서 분석
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), 
                 padding='valid',
                 activation='relu')) #(7필터 2,2 ) / (None, 3, 3, 7) 
model.add(Flatten())  #(N, 63)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

