import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron

#데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0, 1 ,1 ,0]


#2.모델
# model = LinearSVC()
model = Perceptron()

#3.훈련
model.fit(x_data, y_data)

#4 평가,예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 :",y_predict)

results = model.score(x_data,y_data)
print("accuracy_score:",results)

acc = accuracy_score(y_data,y_predict)
print("accuracy_score:", acc)