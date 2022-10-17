from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from keras.datasets import mnist,cifar10
import pandas as pd
import numpy as np


#1. 데이터 전처리

x_train = np.load('D:/study_data/_save/_npy/keras49_4_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_4_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_4_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_4_test_y.npy')
print(x_train)

print(x_test.shape)
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(32,32,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(100,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=4,verbose=2,
          validation_split=0.25,batch_size=5000)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
print('y_predict :', y_predict.shape) #(10000, 100)
#증폭 후 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from keras.datasets import mnist,cifar10
import pandas as pd
import numpy as np


#1. 데이터 전처리

x_train = np.load('D:/study_data/_save/_npy/keras49_4_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_4_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_4_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_4_test_y.npy')
print(x_train)

print(x_test.shape)
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(32,32,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(100,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=4,verbose=2,
          validation_split=0.25,batch_size=5000)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
print('y_predict :', y_predict.shape) #(10000, 100)
#증폭 후 
# loss : 4.604773998260498