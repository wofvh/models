from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.models import Sequential,load_model,Model

#1.데이터
datasets = load_breast_cancer()

# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
 
# scaler =  MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068
 
##### [ 3가지 성능 비교 ] #####
# scaler 사용하기 전
# scaler =  MinMaxScaler()
# scaler = StandardScaler()

#2.모델구성
# model = Sequential() #순차적 
# model.add(Dense(6, activation='linear', input_dim=30)) #sigmoid 0~1 로 분류함 0.5 기준으로 (반올림)
# model.add(Dense(100, activation='linear'))
# model.add(Dense(100, activation='linear'))   #sigmoid 사용해보기 
# model.add(Dense(100, activation='sigmoid'))  #relu 히든레이어에서만 가능 
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(1, activation='sigmoid'))
input1 = Input(shape=(30,))
dense1 = Dense(10)(input1)
dense2 = Dense(25,activation='relu')(dense1)
dense2 = Dense(24,activation='relu')(dense2)
dense3 = Dense(35,activation='sigmoid')(dense1)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy','mse'])
from tensorflow.python.keras.callbacks import EarlyStopping 
earlystopping = EarlyStopping(monitor='loss',patience=500,mode='min', verbose=1,
              restore_best_weights=True)

start_time = time.time()

hist = model.fit(x_train,y_train, epochs=1000, batch_size=100, verbose=1,
                 callbacks=[earlystopping], validation_split = 0.2)
end_time = time.time() - start_time           
 
model.save("./_save/keras22_hamsu04_cancer.h5")
#평가,예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)


y_predict = model.predict(x_test)
y_predict = y_predict.round(0)

#######[과제 accuracy_score 완성]###########
acc= accuracy_score(y_test, y_predict)
print('acc_score:', acc)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2 )
# print("걸린시간:", end_time )

#StandardScaler()
# acc_score: 0.9473684210526315
# r2스코어 : 0.7714667557634479
# 걸린시간: 56.53335666656494