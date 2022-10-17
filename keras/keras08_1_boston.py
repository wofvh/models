
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score 

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.85, shuffle=True, random_state=1234)

print (x)
print (y)
print(x.shape, y.shape)   #(506. 13) (506,)

print(datasets.feature_names)
print(datasets.DESCR)

# # [실습] 아래를 완성하기
# 1. train 0.7
# 2. R2 0.8 이상


# 2. 모델구성

model = Sequential()
model.add(Dense(516, input_dim=13))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='mae', optimizer='adam')

model.fit(x_train, y_train, epochs = 150,
          batch_size=100, verbose=1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


#R2
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# loss :  6.1555399894714355
# r2스코어 : 0.40817593250292994