from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################
import time
start = time.time()
from tensorflow.python.keras.layers import Reshape


#1. 데이터

import numpy as np
x1_datasets = np.array([range(100), range(301,401)]) # 삼성전자 종가, 하이닉스 종가
x1 = np.transpose(x1_datasets) 
print(x1.shape) # (100, 2) (100, 3)

y1 = np.array(range(2001,2101)) # 금리 (100, )
y2 = np.array(range(201,301)) # 금리 (100, )

from sklearn.model_selection import train_test_split

x1_train, x1_test, \
y1_train, y1_test, y2_train, y2_test = train_test_split(x1,y1,y2,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

print(x1_train.shape, x1_test.shape) # (80, 2) (20, 2)
print(y1_train.shape, y1_test.shape) # (80,) (20,)
print(y2_train.shape, y2_test.shape) # (80,) (20,)


#2. 모델
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='ak1')(input1)
dense2 = Dense(200, activation='relu', name='ak2')(dense1)
dense3 = Dense(300, activation='relu', name='ak3')(dense2)
output1 = Dense(100, activation='relu', name='out_ak1')(dense3)
from tensorflow.python.keras.layers import concatenate, Concatenate # 앙상블모델

# concatnate
# merge1 = concatenate([output1, output2, output3], name='mg1')

merge2 = Dense(20, activation='relu', name='mg2')(output1)
merge3 = Dense(300, name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)

#2-4 output 모델1
output41 = Dense(10)(last_output1)
output42 = Dense(10)(output41)
last_output2 = Dense(1)(output42)

#2-4 output 모델2
output51 = Dense(10)(last_output1)
output52 = Dense(10)(output51)
output53 = Dense(10)(output52)
last_output3 = Dense(1)(output53)



model = Model(inputs=input1, outputs=[last_output2, last_output3])


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x1_train, [y1_train, y2_train], epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
# model.summary()
# # loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
# loss_1 = model.evaluate(x1_test, y1_test)
# loss_2 = model.evaluate(x1_test, y2_test)
# print(x1_test.shape)
# print(y1_test.shape, y2_test.shape)

# y1_predict, y2_predict = model.predict(x1_test)
# print(y1_predict)
# print(y2_predict)
# print(y1_test)
# print(y2_test)
# # y_predict = np.argmax(y_predict, axis= 1)
# # y_predict = to_categorical(y_predict)


# r2_1 = r2_score(y1_test, y1_predict)
# r2_2 = r2_score(y2_test, y2_predict)
# print('loss1 : ', loss_1)
# print('loss1 : ', loss_2)
# print('r2스코어1 : ', r2_1)
# print('r2스코어2 : ', r2_2)
# # print('결과값 : ', y_predict)
# print("time :", time.time() - start)

# # loss1 :  [3239872.5, 0.07074783742427826, 3239872.5, 0.175140380859375, 1799.9644775390625] 
# # loss2 :  [3239772.5, 3239772.5, 0.07906484603881836, 1799.9368896484375, 0.20821762084960938]
# # r2스코어1 :  0.9999104828806418
# # r2스코어2 :  0.9998999593831789
# # time : 56.027087450027466

# ##########################################################################
loss = model.evaluate(x1_test, [y1_test, y2_test])
y_predict = model.predict(x1_test)
y_predict = np.array(y_predict) #(2, 20, 1)
print(y_predict) 
print(np.array([y1_test, y2_test]))
y_test=np.array([y1_test, y2_test])
print(y_test.shape) 
y_predict = y_predict.reshape(2, 20)
y_test = y_test.reshape(2, 20)
print(y_test.shape) 

r2 = r2_score(y_test, y_predict)
print('loss: ', loss)
print('r2스코어 : ', r2)
# print('결과값 : ', y_predict)
print("time :", time.time() - start)
