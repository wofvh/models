from keras.datasets import mnist 
from tensorflow.python.keras.models import Sequential
from warnings import filters
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#데이터1 
# model = Sequential
 
#  model.add(Dense(64, input_shape = (28*28)))
# model.add(Dense(64, input_shape = (784,)))

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

###################리세이프#######################
x_train = x_train.reshape(60000, 28* 28* 1)
x_test = x_test.reshape(10000, 28* 28* 1)

print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################

# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

####################스케일러#############################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# scaler.fit(x_train) 
# # scaler.transform(x_test)
# x_test =scaler.transform(x_test)
# x_train = scaler.transform(x_train)



# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)

###################겟더미#######################
# y = pd.get_dummies(y)  #겟더미는 y_predict 할때 np아니고 tf.argmax로 바꾸기
# print(y)
y_train = pd.get_dummies((y_train))
y_test = pd.get_dummies((y_test))
###############################################

# ####################원핫인코더###################
# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)
# ################################################

###################케라스########################
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))   # y의 라벨값 :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(x_train.shape, y_train.shape) 
# print(x_test.shape, y_test.shape)
################################################


# 맹그러바바바
# acc 0.98이상
# cifar는 칼라 패션은 흑백


#2. 모델구성
model = Sequential()
# model.add(Dense(units=64, input_shape = (3,)))         #  (batch_size, input_dim)             input_shape = (10, 10, 3)
# model.summary()
# (input_dim + bias) * units = summary Param # (Dense 모델)
# model.add(Conv2D(filters = 64, kernel_size=(3,3), # kernel_size = 이미지 분석을위해 2x2로 잘라서 분석하겠다~
#                  padding='same', # padding : 커널 사이즈대로 자르다보면 가생이는 중복되서 분석을 못해주기때문에 행렬을 키워주는것, 패딩을 입혀준다? 이런 너낌
#                  input_shape=(28,28,1))) #  (batch_size, rows, columns, channels)            conv2d : model.add input_shape= (x, y, z) x=가로 픽셀 y=세로픽셀 z= 컬러 흑백

model.add(Dense(10, input_shape = (784,)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), 
#                  padding='valid', # 디폴트 값 32 
#                  activation='relu'))
# model.add(Conv2D(32, (2,2), 
#                  padding='valid', # 디폴트 값
#                  activation='relu'))
model.add(Flatten())  # (N, 5408)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

# save_filepath = './_ModelCheckPoint/' + current_name + '/'
# load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')


# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath= "".join([save_filepath, date, '_', filename])
#                       )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1000,
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


# acc스코어 :  0.9343