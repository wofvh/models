from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier,VotingRegressor
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#.1데이터
datasets = load_breast_cancer()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df.head(7))  #head는 기본 5개 출력

x_train ,x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, random_state=123, train_size=0.8, shuffle=True, stratify=datasets.target)



Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)

#voting 은 hard &soft가있음 #estimators= 두개이상은 리스트로 넣어줘야함
model = VotingClassifier(estimators=[('LR', lr), ('KNN', knn)], voting='soft') #함수 voting은 회기모델에서는 안 넣어줌 

#3. 평가예측
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)
print(model.score(x_test,y_test))

score = accuracy_score(y_test,y_predict)
print("보팅결과 : ", round(score,4 ))

# 보팅결과 :  0.9912

classifier = [lr, knn]

for model2 in classifier:  #model2는 모델이름 # 
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_predict)
    class_name = model2.__class__.__name__  #<모델이름 반환해줌 
    print("{0}정확도 : {1:.4f}".format(class_name, score2)) # f = format