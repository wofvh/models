
from sklearn.datasets import load_boston, fetch_covtype
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 

from sklearn.pipeline import make_pipeline

#1.데이터
datasets = fetch_covtype()
x,y = datasets.data, datasets.target
print(x.shape,y.shape)  #(150, 4) (150,)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=1234, train_size=0.8,)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

model = make_pipeline(StandardScaler(),LinearRegression())

model.fit(x_train,y_train)
print("기본 스코어 : ", model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold, scoring= 'r2')
print("cv:",scores)
print("기냥cv 엔빵:",np.mean(scores))

#2.모델구성

############################PolynomialFeatures후 ########################################

pf = PolynomialFeatures(degree=2, include_bias = False) #차수는 2로 설정
# pf = PolynomialFeatures(degree=2,) #차수는 2로 설정
xp = pf.fit_transform(x) 
print(xp.shape)  #(506, 105)

x_train,x_test,y_train,y_test = train_test_split(xp,
                                                 y,random_state=1234, train_size=0.8,)

# 2.모델구성
model = make_pipeline(StandardScaler(),LinearRegression())

model.fit(x_train,y_train)

print("폴리 스코어 : ", model.score(x_test, y_test))

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 나눈 값 : ", np.mean(scores))




# (581012, 54) (581012,)
# 기본 스코어 :  0.31613448577948156
# cv: [0.32106792 0.32130745 0.31817901 0.31705079 0.32011629]
# 기냥cv 엔빵: 0.31954429186301087
# (581012, 1539)
# 폴리 스코어 :  -2.3158981622457397e+21
# 폴리 CV :  [ 4.84561122e-01 -1.28227594e+21  4.79621355e-01  4.83901721e-01
#  -4.65498378e+21]
# 폴리 CV 나눈 값 :  -1.1874519425780437e+21