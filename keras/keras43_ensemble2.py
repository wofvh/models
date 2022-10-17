#1.데이터 
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201),range(411, 511),range(150,250)])# 원유. 돈육 ,밀
x3_datasets = np.array([range(100, 200),range(1301, 1401)])# 원유. 돈육 ,밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape)  #(100, 2) (100,3)

y = np.array(range(2001, 2101)) #금리  print(y.shape)#(100,) 

from sklearn.model_selection import train_test_split

x1_train , x1_test , x2_train, x2_test, x3_train,x3_test, y_train, y_test = train_test_split(x1, x2 ,x3,y ,train_size=0.7,
                                                                            random_state=6 )

print(x1_test.shape, x1_train.shape)    #(30, 2) (70, 2)
print(x2_test.shape, x2_train.shape)    #(30, 3) (70, 3)
print(x3_test.shape, x3_train.shape)    #(70, 2) (30, 2)
print(y_test.shape, y_train.shape)      #(30,) (70,)    

#2.모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Input

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(75, activation='relu', name= 'ys1')(input1)
dense2 = Dense(65, activation='relu', name= 'ys2')(dense1)
dense3 = Dense(58, activation='relu', name= 'ys3')(dense2)
output1 = Dense(55, activation='relu', name= 'out_ys1')(dense3)

#2-2,모델
input2 = Input(shape=(3,))
dense11 = Dense(31, activation='relu', name= 'ys11')(input2)
dense12 = Dense(56, activation='relu', name= 'ys12')(dense11)
dense13 = Dense(36, activation='relu', name= 'ys13')(dense12)
dense14 = Dense(38, activation='relu', name= 'ys14')(dense13)
output2 = Dense(30, activation='relu', name= 'out_ys2')(dense14)

input3 = Input(shape=(2,))
dense21 = Dense(31, activation='relu', name= 'ys21')(input3)
dense22 = Dense(56, activation='relu', name= 'ys22')(dense21)
dense23 = Dense(36, activation='relu', name= 'ys23')(dense22)
dense24 = Dense(38, activation='relu', name= 'ys24')(dense23)
output3 = Dense(30, activation='relu', name= 'out_ys3')(dense24)

from tensorflow.python.keras.layers import concatenate, Concatenate
mergel = concatenate([output1, output2,output3], name = 'mg1')
merge2 = Dense(2, activation='relu', name='mge2')(mergel)
merge3 = Dense(3, name ='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs = [input1, input2, input3], outputs = last_output)

model.summary()

#컴파일 , 훈련 
model.compile(loss = 'mae',optimizer='adam')

hist = model.fit([x1_test, x2_test, x3_test],y_train ,epochs=10, batch_size=100,
                 validation_split=0.2,
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test],y_test) 
# y_predict = model.predict(x1_test, x2_test)
print('loss : ', loss)

y_predict = model.predict([x1_test, x2_test, x3_test])

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
