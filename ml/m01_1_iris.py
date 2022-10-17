import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tensorboard import summary
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.svm import LinearSVC
from sklearn import datasets

import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets['data']
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(150, 4) (150,)

print("y의 라벨값:" , np.unique(y)) #y의 라벨값: [0 1 2]

print(y)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, shuffle= True,
                                                    random_state=66 )
print(y_train)
print(y_test)

scaler = RobustScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


model = LinearSVC()

#3. 훈련
model.fit(x_train, y_train)


#4.평가,예측
results = model.score(x_test,y_test)

print('결과 acc:', results)

#회기모델 r2 스코어 