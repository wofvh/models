from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense,Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston
import numpy as np
import inspect, os
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import time
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=100
                                                    )
print(x_train.shape,x_test.shape) #(354, 13) (152, 13) 


x_train = x_train.reshape(354, 13,1,1)
x_test = x_test.reshape(152, 13,1,1)
print(x_train.shape)

print(np.unique(y_train, return_counts=True))
# (array([ 5. ,  5.6,  7. ,  7.2,  7.4,  7.5,  
# 8.3,  8.4,  8.8,  9.6,  9.7,

#####################XXXXX스케일러XXXXX######################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
# model.add(Dense(80, input_dim=(354,13,1,1)))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.summary()
model.add(Conv2D(filters = 200, kernel_size=(1,1), # kernel_size = 이미지 분석을위해 2x2로 잘라서 분석하겠다~
                 padding='same', # padding : 커널 사이즈대로 자르다보면 가생이는 중복되서 분석을 못해주기때문에 행렬을 키워주는것, 패딩을 입혀준다? 이런 너낌
                 input_shape=(13,1,1))) #  (batch_size, rows, columns, channels)            conv2d : model.add input_shape= (x, y, z) x=가로 픽셀 y=세로픽셀 z= 컬러 흑백
model.add(Conv2D(64, (3,3), 
                 padding='same', # 디폴트 값
                 activation='relu'))
model.add(Flatten())  # (N, 5408)
model.add(Dense(151, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print(x_test.shape,y_predict.shape)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

# dropout 작용후 
# loss :  15.32930850982666
# r2스코어 :  0.8165976217640427        
# 걸린시간 :  8.414448976516724

# dropout 적용후
# loss :  19.219741821289062
# r2스코어 :  0.7700518862054726        
# 걸린시간 :  8.179402828216553

