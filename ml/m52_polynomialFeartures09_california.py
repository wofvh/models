from sklearn.datasets import load_boston, fetch_california_housing
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
datasets = fetch_california_housing()
x,y = datasets.data, datasets.target
print(x.shape,y.shape)  #(150, 4) (150,)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=1234, train_size=0.8,)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2.모델구성
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
# np.delete(scores[0],[3])

scores = np.delete(scores, [0],[2], axis=0)
print("폴리 CV : ", scores)
print("폴리 CV 나눈 값 : ", np.mean(scores))

# 기본 스코어 :  0.6065722122106435
# cv: [0.6153114  0.59856063 0.61434956 0.59191848 0.58489428]
# 기냥cv 엔빵: 0.6010068700547343
# (20640, 44)
# 폴리 스코어 :  0.5005165687196428
# 폴리 CV :  [-231.1907676     0.62402233   -3.94169064    0.60985177    0.66558945]
# 폴리 CV 나눈 값 :  -46.6465989383862