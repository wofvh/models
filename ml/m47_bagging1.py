from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,f1_score

datasets = load_breast_cancer()
x, y =datasets.data, datasets.target

print(x.shape, y.shape)   #(569, 30) (569,)

x_train , x_test, y_train, y_test = train_test_split(
    x,y, random_state=123, train_size=0.8, shuffle=True, stratify=y)

Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)


# Bootstrap Aggregation 
#Bagging 한가지 모델을 여러번 돌려서 사용 
from sklearn.ensemble import BaggingClassifier #Bagging 앙상블 모델엣 가장많이 사용함 
from sklearn.linear_model import LogisticRegression

model = BaggingClassifier(LogisticRegression(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=123,
                          )

model.fit(x_train,y_train)

# 평가예측

print(model.score(x_test,y_test))
print("score",)
