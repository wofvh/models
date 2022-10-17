from posixpath import split
import numpy as np

from sklearn import datasets 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM,GRU

a = np.array(range(1,101)) #(1~~100
x_predict = np.array(range(96, 106))
size1 = 5                               #<<< 이럴경우 x5 개 y는 1 개

###########시계열 데이터####원하는 y 값을 만들어준다 
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):  
        subset = dataset[i : (i + size)]
        aaa.append(subset)       
    return np.array(aaa)

bbb = split_x(a, size1)
print(bbb.shape)  #(96, 5)

x = bbb[:, :-1]
y = bbb[:,  -1:]

# k = ccc[:, -1]
print(x)
print(y.shape) #(96, 4) (96,1)
# print(k)

# x = x.reshape(96,4,1)
# y = y.reshape(96,1,1)
print(x_predict.shape)


#모델구성
model = Sequential()                
model.add(Dense(64,input_dim=4, activation='relu')) 
model.add(Dense(36, activation='relu'))    
model.add(Dense(33, activation='relu'))    
model.add(Dense(37, activation='relu'))    
model.add(Dense(39, activation='relu'))    
model.add(Dense(38, activation='relu'))    
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss ='mse',optimizer='adam')
model.fit(x,y, epochs=100)

# from tensorflow.python.keras.callbacks import EarlyStopping 
# earlystopping = EarlyStopping(monitor='loss',patience=20, mode='min', verbose=1,
#               restore_best_weights=True)

#4. 평가 예측
loss = model.evaluate(x,y)
ccc = split_x(x_predict,size1)
z = ccc[:, :-1]

# z = z.reshape(6, 4, 1)  
print(x_predict)        #[ 96  97  98  99 100 101 102 103 104 105]
print(x_predict.shape) #(10,)
result = model.predict(z)  #평가예측에서 똑같이 맟춰서
print('loss:',loss)
print('1~100 의 결과:',result)  #RNN input_shape 에서 들어간 차원을 야함
