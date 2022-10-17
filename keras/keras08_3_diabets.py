from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential#순차적 코드진행
from tensorflow.python.keras.layers import Dense#노드를 연결해줌 
from sklearn import datasets
from sklearn.datasets import load_diabetes #sklearn 데이터 저장소

#데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=1234) 

# print(x) 
# print(y)
# print(x.shape, y.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)


#1.모델
model = Sequential() #순차적 
model.add(Dense(516, input_dim=10))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))


#2.컴파일 훈련model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=380, batch_size=18)

#3.평가 예측
loss = model.evaluate(x_test, y_test) #평가값을 로스에 넣는다 #(evaliate 평가) 
print('loss : ', loss)

y_predict = model.predict(x_test) #
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


#결과 loss :  2225.1044921875
#r2스코어 : 0.6285839988523811
