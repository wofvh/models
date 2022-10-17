#1. r2(결정개수) 를 음수가 아닌 0.5 이하로 만들것
#2. 데이터 건들지 않기
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. 훈련 100번 이상
#7. train 70%
#8. loss 지표는 mse, ,mae
 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,8,5,16,17,23,21,20])
from posixpath import split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123)   #shuffle을 바꿧을때도 효과있음 fleas or true

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100
          , batch_size=1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 :', r2 )


# loss :  20.037099838256836
# r2스코어 : 0.792691770992876