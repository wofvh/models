from tabnanny import verbose
from tracemalloc import start
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score 


#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=1234)

#print (x)
#print (y)
#print(x.shape, y.shape)   #(506. 13) (506,)

#print(datasets.feature_names)
#print(datasets.DESCR)

#[실습] 아래를 완성하기
# 1. train 0.7
# 2. R2 0.8 이상


#2.모델구성
model = Sequential() #순차적 
model.add(Dense(512, input_dim=13))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

import time
#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
print(start_time) #1656033141.8530962

model.fit(x,y, epochs=50
          , batch_size=1, verbose=1)
end_time = time.time() - start_time

print("걸린시간:", end_time = start_time)

'''
verbose 0 걸린시간 #걸린시간: 1656570737.0593293 
verbose 1 걸린시간 #걸린시간: 18.045161247253418 
verbose 2 걸린시간 #걸린시간: 18.045161247253418 
verbose 3.4~ 걸린시간 #걸린시간: 18.045161247253418
'''

 