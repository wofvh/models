import numpy as np
from sqlalchemy import false 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
# 출처: https://rfriend.tistory.com/519 [R, Python 분석과 프로그래밍의 친구 (by R Friend):티스토리]

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_train = np.array([1,2,3,4,5,6,7])
# y_test = np.array([8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,                                                   
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
print(x_train) #[2 7 6 3 4 8 5]
print(x_test) #[ 1  9 10]
print(y_train) #[2 7 6 3 4 8 5]
print(y_test) #[ 1  9 10]


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result) 


# 11의 예측값 :  [[10.994131]]