from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.datasets import load_diabetes
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.models import Sequential,load_model,Model

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

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

#2. 모델구성
# model = Sequential() #순차적 
# model.add(Dense(9, input_dim=10))
# model.add(Dense(19, activation='sigmoid'))
# model.add(Dense(19, activation='relu'))
# model.add(Dense(22, activation='relu'))
# model.add(Dense(19, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(1))
input1 = Input(shape=(10,))
dense1 = Dense(19)(input1)
dense2 = Dense(23,activation='relu')(dense1)
dense3 = Dense(23,activation='relu')(dense2)
dense2 = Dense(23,activation='relu')(dense3)
dense3 = Dense(32,activation='sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,verbose=1,
                 validation_split=0.2, callbacks=[earlyStopping])

end_time = time.time()

model.save("./_save/keras22_hamsu03_diabets.h5")
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

#StandardScaler()
# loss: [0.2693632245063782, 0.9912280440330505, 0.010915878228843212]
# acc_score: 0.9912280701754386
# r2스코어 : 0.9619111259605747
# 걸린시간: 56.982468128204346