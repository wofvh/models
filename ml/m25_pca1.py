from unittest import result
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__)
import warnings
warnings.filterwarnings(action="ignore")

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target
print(x.shape,y.shape )    #(506, 13) (506,)

pca = PCA(n_components=2)
x = pca.fit_transform(x)
print(x.shape)              #(506, 2)



x_train , x_test, y_train, y_test = train_test_split(
    x,y, train_size= 0.8, random_state=123 ,shuffle=True
)


from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 

model = RandomForestRegressor()

model.fit(x_train,y_train)

#3.훈련 
results =  model.score(x_test,y_test)      #(x_train,y_train, eval_metric= 'error')
print('결과:', results)