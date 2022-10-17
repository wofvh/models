from json import load
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_boston, fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
    )

Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델
from sklearn.ensemble import VotingClassifier,VotingRegressor
from xgboost import XGBClassifier,XGBRFRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
xg = XGBRFRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor() #(verbose=False)#catboost vervose가 많음 ! 그래서 다른모델이랑 성능비교 시에는 주석처리

#voting 은 hard &soft가있음 #estimators= 두개이상은 리스트로 넣어줘야함
model = VotingRegressor(estimators=[('xg', xg), ('cat', cat),("lg", lg)]) 

#3. 평가예측
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)
print(model.score(x_test,y_test))

score = r2_score(y_test,y_predict)
print("보팅결과 : ", round(score,4 ))

# 보팅결과 :  0.9912

classifier = [cat,xg, lg,]

for model in classifier:  #model2는 모델이름 # 
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    score2 = r2_score(y_test,y_predict)
    class_name = model.__class__.__name__  #<모델이름 반환해줌 
    print("{0}정확도 : {1:.4f}".format(class_name, score2)) # f = format
    
print("보팅결과 : ", round(score,4 ))
    
# CatBoostRegressor정확도 : 0.7958
# XGBRFRegressor정확도 : 0.7610
# LGBMRegressor정확도 : 0.7679
# 보팅결과 :  0.7911