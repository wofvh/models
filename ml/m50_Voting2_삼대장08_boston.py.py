import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing,load_boston
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier,XGBRFRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings(action="ignore")

#.1데이터
datasets = load_boston()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df.head(7))  #head는 기본 5개 출력

x_train ,x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, random_state=123, train_size=0.8, shuffle=True,)



Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델
xg = XGBRFRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor() #(verbose=False)#catboost vervose가 많음 ! 그래서 다른모델이랑 성능비교 시에는 주석처리

#voting 은 hard &soft가있음 #estimators= 두개이상은 리스트로 넣어줘야함
model = VotingRegressor(estimators=[('xg', xg), ('cat', cat),("lg", lg)], ) 

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
    
# XGBClassifier정확도 : 0.9912

# CatBoostRegressor정확도 : 0.8763
# XGBRFRegressor정확도 : 0.7917
# LGBMRegressor정확도 : 0.7481
# 보팅결과 :  0.8268