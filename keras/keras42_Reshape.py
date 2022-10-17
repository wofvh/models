from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM,Conv1D,Reshape

#1. 데이터
(x_train, y_train), (x_test, y_test) =mnist.load_data()

print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28,1)  # input 28,28,1 
x_test = x_test.reshape(10000, 28, 28,1)    # 

print(x_train.shape)
print(np.unique(y_train, return_counts =True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

y_train = pd.get_dummies((y_train))
y_test = pd.get_dummies((y_test))
print(x_train.shape)
# 실습 acc 0.98이상 
# 원핫인코딩 

#2. 모델구성 

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3),   # 출력(4,4,10)                                    
                 padding='same',
                 input_shape=(28, 28, 1)))                                                                
model.add(MaxPooling2D())                   #(N,14,14,64)
model.add(Conv2D(32,(3,3)))                 #(N,12,12,32)
model.add(Conv2D(7,(3,3)))                  #(N,10,10,7)
model.add(Flatten())                        #(N,700)
model.add(Dense(100, activation= 'relu'))   #(N,100)
model.add(Reshape(target_shape=(100,1)))    #(N,100,1)
model.add(Conv1D(10, kernel_size =(3,)))    #(N,98,10)
model.add(LSTM(16))                         #(N,16)
model.add(Dense(32, activation= 'relu'))    #(N,32)
model.add(Dense(10, activation= 'softmax')) #(N,10)

model.summary()



#3. 컴파일 구성 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
              verbose=1, restore_best_weights = True)     
        
hist = model.fit(x_train, y_train, epochs=50, batch_size=1000,verbose=1,
                 validation_split=0.2, callbacks=[earlystopping])


# model.save("./_save/keras23_9_load_diabet.h5")
# model = load_model("./_save/keras23_9_load_diabet.h5")

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)


# (kernel_size * channls + bias) * filters(output) = summary Param 개수

# tf.keras.layers.Dense(
#     units,                                # output 로드 개수 10 
#     activation=None,
#     use_bias=True,                        # 
#     kernel_initializer="glorot_uniform",  # 레이어 초기화
#     bias_initializer="zeros",             # 레이어 초기화
#     kernel_regularizer=None,              # 정규화, 규제화 
#     bias_regularizer=None,                # 정규화, 규제화 
#     activity_regularizer=None,            # 정규화, 규제화 
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs)

#      model.add(Dense(10, activation ='relu', input_dim =8)
#      2차원일때 input shape ) Dense > (batch_size(행),input_dim(열))
