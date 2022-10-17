import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
import time

from sklearn.preprocessing import MaxAbsScaler, RobustScaler  #(두개의 차이점 알아보기)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_boston()
x = datasets.data 
y = datasets.target

# print(np.min(x))  # x의 최소값이 출력된다.
# #   x의 최소값 = 0.0 
# print(np.max(x))  # x의 최소값이 출력된다.
# #   x의 최대값 = 711.0

# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# # MinMaxScaler 표준편차 공식: [(x - 최소값) /(나누기) 최대값 - 최소값]
# # 위에 공식대로 해야 1이 나온다.
# print(x[:10])
 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    random_state=66,
                                                    )
scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # 0
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 1.0000000000000002
print(np.min(x_test))   # -0.06141956477526944
print(np.max(x_test))   # 1.1478180091225068
 
##### [ 3가지 성능 비교 ] #####
# scaler 사용하기 전
# scaler =  MinMaxScaler()
# scaler = StandardScaler()


 #2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=13))
model.add(Dense(19, activation='sigmoid'))
model.add(Dense(19, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1))



#3. 컴파일. 훈련

model.compile(loss='mae', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=800, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   
print('loss : ', loss)
print('r2스코어 : ', r2)

#StandardScaler()
# 걸린시간 :  1657089139.756187
# loss :  2.4011130332946777
# r2스코어 :  0.8471342166747788



 
 
 
 
 

 
 