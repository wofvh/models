from gc import callbacks
from tabnanny import verbose
import numpy as np
from tracemalloc import start
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7, shuffle=True,
                                                    random_state=50 )

#print (x)
#print (y)
#print(x.shape, y.shape)   #(506. 13) (506,)

#print(datasets.feature_names)
#print(datasets.DESCR)
# scaler =  MinMaxScaler()

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068


#2.모델구성
model = Sequential() #순차적 
model.add(Dense(6, input_dim=13))
model.add(Dense(28,activation='relu'))
model.add(Dense(52,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.summary()

# # model.save("./_save/keras23_3_save_model.h5")
# model.save_weights("./_save/keras23_5_save_weights1.h5")

# # model = load_model("./_save/keras23_3_save_model.h5")
model.load_weights('./_save/keras23_5_save_weights1.h5')
model.load_weights('./_save/keras23_5_save_weights2.h5')


# import time
#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')


# from tensorflow.python.keras.callbacks import EarlyStopping 
# earlystopping = EarlyStopping(monitor='loss',patience=100, mode='min', verbose=1,
#               restore_best_weights=True)
# hist = model.fit(x,y, epochs=14, batch_size=100, verbose=1,
#                  callbacks=[earlystopping],
#                  validation_split = 0.2)

# # model.save("./_save/keras23_3_save_model.h5")

# model = load_model("./_save/keras23_3_save_weights2.h5")

#평가,예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('r2스코어:',r2)
print('loss', loss)
