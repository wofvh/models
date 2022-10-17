from unittest import result
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing ,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__)
import warnings
warnings.filterwarnings(action="ignore")

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape,y.shape )    #(569, 30) (569,)


pca  = PCA(n_components=15)  
x = pca.fit_transform(x)
print(x.shape)              #(506, 2)



for i in range(1, 31) :
#    x = datasets.data
    pca = PCA(n_components=i)
    x2 = pca.fit_transform(x)
    print(i, "번 압축했을때 Shape : " , x2.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=666, shuffle=True
    )
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print('결과 : ', result)
    print("="*40)


x_train , x_test, y_train, y_test = train_test_split(
    x,y, train_size= 0.8, random_state=123 ,shuffle=True
)


from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 

model = RandomForestClassifier()

model.fit(x_train,y_train)

#3.훈련 
results =  model.score(x_test,y_test)      #(x_train,y_train, eval_metric= 'error')
print('결과:', results)

# (569, 15)
# 결과: 0.9912280701754386

for i in range(1, 31) :
#    x = datasets.data
    pca = PCA(n_components=i)
    x2 = pca.fit_transform(x)
    print(i, "번 압축했을때 Shape : " , x2.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=666, shuffle=True
    )
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print('결과 : ', result)
    print("="*40)
    