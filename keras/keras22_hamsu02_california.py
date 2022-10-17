from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import time 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from tensorflow.python.keras.models import Sequential,load_model,Model

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test)
# scaler =  MinMaxScaler()

#2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=8,activation='sigmoid'))
# model.add(Dense(13,activation='relu'))
# model.add(Dense(14,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(14,activation='relu'))
# model.add(Dense(12))
# model.add(Dense(1))
input1 = Input(shape=(8,))
dense1 = Dense(21)(input1)
dense2 = Dense(45,activation='relu')(dense1)
dense3 = Dense(51,activation='sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

model.save("./_save/keras22_hamsu02_california.h5")

print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

#StandardScaler()
# 걸린시간 :  1657090622.6826775
# loss :  0.3365107774734497
# r2스코어 :  0.7994494457704042