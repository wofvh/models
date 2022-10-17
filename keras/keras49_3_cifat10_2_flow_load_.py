from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from keras.datasets import mnist,cifar10
import pandas as pd
import numpy as np


#1. 데이터 전처리
# np.save('D:/study_data/_save/_npy/keras49_3_train_x.npy',arr=xy_df3[0][0])
# np.save('D:/study_data/_save/_npy/keras49_3_train_y.npy',arr=xy_df3[0][1])
# np.save('D:/study_data/_save/_npy/keras49_3_test_x.npy',arr=x_test)
# np.save('D:/study_data/_save/_npy/keras49_3_test_y.npy',arr=y_test)

x_train = np.load('D:/study_data/_save/_npy/keras49_3_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_3_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_3_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_3_test_y.npy')

print(x_train.shape) #(40000, 32, 32, 3)
print(x_test.shape) #(10000, 32, 32, 3)
print(y_train.shape) #(40000, 1)
print(y_test.shape)# (40000, 1)

#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(32,32,3),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,verbose=2,validation_split=0.25,batch_size=500)
# hist = model.fit_generator(x_train,y_train,epochs=2,
#                     validation_split=0.25,
#                     steps_per_epoch=32,
#                     validation_steps=4) # 배치가 최대 아닐 경우 사용

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
print('y_predict :', y_predict)

#증폭 후 
# loss : [1.3156336545944214, 0.5347999930381775]