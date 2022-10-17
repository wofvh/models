#1.데이터 
import numpy as np
import tensorflow as tf
x1_datasets = np.array([range(100), range(301, 401)]) #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201),range(411, 511),range(150,250)])# 원유. 돈육 ,밀
x3_datasets = np.array([range(100, 200),range(1301, 1401)])# 원유. 돈육 ,밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape)     #(100, 2)
print(x2.shape)     #(100, 3)
print(x3.shape)     #(100, 2)


print(x1.shape, x2.shape, x3.shape)  #(100, 2) (100, 3) (100, 2)

y1 = np.array(range(2001, 2101)) #금리  print(y.shape)#(100,) 
y2 = np.array(range(201, 301)) #금리  print(y.shape)#(100,) 

from sklearn.model_selection import train_test_split

x1_train , x1_test , x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, \
    y2_trian, y2_test= train_test_split(x1, x2 ,x3, \
        y1, y2,train_size=0.8,random_state=6,shuffle=True )

print(x1_test.shape, x1_train.shape)    #(20, 2) (80, 2)
print(x2_test.shape, x2_train.shape)    #(20, 3) (80, 3)
print(x3_test.shape, x3_train.shape)    #(20, 2) (80, 2)
print(y1_test.shape, y1_train.shape)      #(20,) (80,)    
print(y2_test.shape, y2_trian.shape)      #(20,) (80,)    
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
#2-3,모델
input3 = Input(shape=(2,))
dense21 = Dense(31, activation='relu', name= 'ys21')(input3)
dense22 = Dense(56, activation='relu', name= 'ys22')(dense21)
dense23 = Dense(36, activation='relu', name= 'ys23')(dense22)
dense24 = Dense(38, activation='relu', name= 'ys24')(dense23)
output3 = Dense(30, activation='relu', name= 'out_ys3')(dense24)

# Concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate
mergel = Concatenate(axis=1)([output1, output2,output3])
#  tf.keras.layers.Concatenate(axis=1)([x, y])
merge2 = Dense(32, activation='relu', name='mge2_1')(mergel)
merge3 = Dense(16, name ='mg3')(merge2)
merge4 = Dense(16, name ='mg4')(merge3)
merge5 = Dense(16, name ='mg5')(merge4)
last_output1 = Dense(1, name='last1')(merge5)

# #2-4. oupput모델1
# output41 = Dense(10)(last_output1)
# output42 = Dense(25)(output41)
# output43 = Dense(26)(output42)
# output44 = Dense(27)(output43)
# last_output2 = Dense(1)(output44)
 
# #2-4. oupput모델2
# output51 = Dense(51)(last_output1)
# output52 = Dense(26)(output51)
# output53 = Dense(25)(output52)
# output54 = Dense(28)(output53)
# output55 = Dense(20)(output54)
# last_output3 = Dense(1)(output55)

merge4 = Dense(64, activation='relu', name='mge2_2')(mergel)
merge5 = Dense(32, name ='mg3_2')(merge4)
merge6 = Dense(32, name ='mg3_3')(merge5)
merge7 = Dense(32, name ='mg3_4')(merge6)
last_output2 = Dense(1, name='last2')(merge7)

model = Model(inputs = [input1, input2, input3], outputs = [last_output1,last_output2,])
model.summary()

#컴파일 , 훈련 
model.compile(loss = 'mae',optimizer='adam')

hist = model.fit([x1_test, x2_test, x3_test],[y1_train, y2_trian],
                 epochs=500, batch_size=50,
                 validation_split=0.2,
                 verbose=1)

#4. 평가, 예측
loss_1 = model.evaluate([x1_test, x2_test, x3_test],y1_test) 
loss_2 = model.evaluate([x1_test, x2_test, x3_test],y2_test) 
# y_predict = model.predict(x1_test, x2_test)
print('loss(y1) : ', loss_1)
print('loss(y2) : ', loss_2)

y1_predic, y2_predict = model.predict([x1_test, x2_test, x3_test])

from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test,y1_predic)
r2_2 = r2_score(y2_test,y2_predict)

print('loss : ' , loss_1)
print('loss : ' , loss_2)
print('r2스코어 : ', r2_1)
print('r2스코어 : ', r2_2)

# r2스코어 :  -0.05386195524101911
# r2스코어 :  -0.008437246798668863