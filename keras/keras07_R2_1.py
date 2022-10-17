from pickletools import optimize
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,8,5,16,17,23,21,20])
from posixpath import split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=66)


print(x_train)            #[18  9  3 15  7 13 12  8  1 19 11 16 14 20]
print(x_test)             #[ 2 17  5  6  4 10]
print(y_train)            #[23  9  3  5  7 13 12  8  1 21 11 16  8 20]
print(y_test)             #[ 2 17  5  6  4 10]
            
#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 :', r2)


# r2스코어 : 0.7995454893349759