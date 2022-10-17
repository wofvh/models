
from sklearn.datasets import load_boston, load_digits
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
datasets = load_digits()
x,y = datasets.data, datasets.target
print(x.shape,y.shape)  #(1797, 64) (1797,1)


x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=1234, train_size=0.8,)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (1437, 64)
# (360, 64)
# (1437,)
# (360,)


kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

model = make_pipeline(StandardScaler(),LogisticRegression())

model.fit(x_train,y_train)
print("기본 스코어 : ", model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold, scoring= 'r2')
print("cv:",scores)
print("기냥cv 엔빵:",np.mean(scores))

#2.모델구성

############################PolynomialFeatures후 ########################################

# pf = PolynomialFeatures(degree=2, include_bias = False) #차수는 2로 설정
pf = PolynomialFeatures(degree=2,) #차수는 2로 설정
xp = pf.fit_transform(x) 
print(xp.shape)  #(506, 105)

x_train,x_test,y_train,y_test = train_test_split(xp,
                                                 y,random_state=1234, train_size=0.8,)

# 2.모델구성
model = make_pipeline(StandardScaler(),LogisticRegression())

model.fit(x_train,y_train)

print("폴리 스코어 : ", model.score(x_test, y_test))

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 나눈 값 : ", np.mean(scores))



# 기본 스코어 :  0.9638888888888889
# cv: [0.88669065 0.91747034 0.9538632  0.94396481 0.8975242 ]
# 기냥cv 엔빵: 0.9199026392965267
# (1797, 2145)
# 폴리 스코어 :  0.9861111111111112
# 폴리 CV :  [0.96964928 1.         0.95885096 0.97417128 0.93671813]
# 폴리 CV 나눈 값 :  0.9678779308190816
