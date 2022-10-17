
import tensorflow as tf
from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)\
print(np.unique(y_train, return_counts=True)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(np.unique(y_test, return_counts=True))



x_train = x_train.reshape(50000, 32* 32* 3)
x_test = x_test.reshape(10000, 32* 32* 3)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

####################겟더미#######################
# y = pd.get_dummies(y)  #겟더미는 y_predict 할때 np아니고 tf.argmax로 바꾸기
# print(y)
################################################

# ####################원핫인코더###################
# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)
# ################################################

###################케라스########################
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))   # y의 라벨값 :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)
################################################
#  dtype=int64))        
# (array([0., 1.], dtype=float32), array([4950000,   50000], dtype=int64))  
# (array([0., 1.], dtype=float32), array([990000,  10000], dtype=int64))    
# (50000, 3072) (50000, 100)
# (10000, 3072) (10000, 100)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_shape = (32*32*3,)))
model.add(Flatten())  # (N, 5408)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.summary()

# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


# save_filepath = './_ModelCheckPoint/' + current_name + '/'
# load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1814_0029-1.3267.hdf5')


earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)      
                    

model.fit(x_train, y_train, epochs=100, batch_size=500,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

print(y_test, y_predict)
print(y_test.shape, y_predict.shape)
acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

# (10000, 10) (10000, 10)
# acc스코어 :  0.5993'


# (10000, 100) (10000, 100)
# acc스코어 :  0.1866