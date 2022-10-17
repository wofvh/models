import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0, 1 ,1 ,1]


#2.모델
# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim=2,))
model.add(Dense(256,activation="relu"))
model.add(Dense(128,))
model.add(Dense(64,))
model.add(Dense(32,))
model.add(Dense(16,))
model.add(Dense(1, activation="relu"))


#3.훈련
model.compile(loss = "binary_crossentropy", optimizer= "adam",metrics=['acc'])
model.fit(x_data, y_data , batch_size=100, epochs=10)


#4 평가,예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 :",y_predict)

results = model.evaluate(x_data,y_data)
print("accuracy_score:",results[1])

# acc = accuracy_score(y_data,y_predict)
# print("accuracy_score:", acc)