import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes


datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test,y_train, y_test = train_test_split(x,y,
                                                   train_size=0.7,
                                                   shuffle=True,
                                                   random_state=100)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.25)

# 평가 예측 
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

#r2
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# loss : 2686.013916015625
# r2스코어 : 0.49314765629535107
