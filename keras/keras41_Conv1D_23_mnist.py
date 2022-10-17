from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling1D, Input, Dropout, Conv1D, MaxPool1D
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
import time
start = time.time()

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

###################리세이프#######################
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################

#####################XXXXX스케일러XXXXX######################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#################################################

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


# 맹그러바바바
# acc 0.98이상
# cifar는 칼라 패션은 흑백


#2. 모델구성
model = Sequential()
# model.add(Dense(units=10, input_shape = (3,)))         #  (batch_size, input_dim)             input_shape = (10, 10, 3)
# model.summary()
# (input_dim + bias) * units = summary Param # (Dense 모델)


input1 = Input(shape=(28,28))
conv2D_1 = Conv1D(100,3, padding='same')(input1)
MaxP1 = MaxPooling1D()(conv2D_1)
conv2D_2 = Conv1D(200,2,
                  activation='relu')(MaxP1)
MaxP2 = MaxPooling1D()(conv2D_2)
flatten = Flatten()(MaxP2)
dense1 = Dense(200)(flatten)
batchnorm1 = BatchNormalization()(dense1)
activ1 = Activation('relu')(batchnorm1)
output1 = Dense(10, activation='softmax')(activ1)
model = Model(inputs=input1, outputs=output1)   

# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=30, batch_size=150,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
print("time :", time.time() - start)

# conv2d
# loss :  [0.04215491563081741, 0.9890000224113464]
# acc스코어 :  0.989

# conv1d
# loss :  [0.12909048795700073, 0.972000002861023]
# acc스코어 :  0.972
# time : 30.388893127441406