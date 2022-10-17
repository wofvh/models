from tracemalloc import start
from unittest import result
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,fetch_covtype,load_digits
from sklearn.datasets import load_breast_cancer,load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(581012, 54) (581012,)



le = LabelEncoder()
y = le.fit_transform(y)

# pca = PCA(n_components=20)   #54 >10
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x,y)
x = lda.transform(x)
print(x)

# pca_EVR = pca.explained_variance_ratio_

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

x_train , x_test, y_train, y_test = train_test_split(
    x,y, train_size= 0.8, random_state=123 ,shuffle=True
)

kfold = KFold(n_splits=5 , shuffle=True, random_state=123) 

parameters = {"n_estimators" : [100,200,300,400,500,1000],
              "learning_rate" : [0.1, 0.2, 0,3, 0.5, 0.01, 0.001],
              'max_depth' : [None,2,3,4,5,6,7,8,9,10],
              'gamma' :[0, 1, 2, 3, 4, 5, 7, 10, 100],
            #   'min_child_weight':[0, 0.01, 0.1, 0.5, 1, 5, 10,100],
            #   'subsample': [0, 0.1, 0.2, 0.3,0.5, 0.7, 1],
            #   'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1],
            #   'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1],
            #   'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1],
            #   'reg_alpha':[0, 0.1, 0.01, 0.001, 1, 2, 10],
              'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10],
              } 


print(np.unique(y_train, return_counts= True))

# (array([1, 2, 3, 4, 5, 6, 7]), array([169507, 226569,  28696,   2152,   7618,  13864,  16403],
#       dtype=int64))

#2. 모델
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb = XGBClassifier(tree_method ='gpu_hist', predictor = 'gpu_predictor', gpu_id = 0,)

model = GridSearchCV(xgb, parameters, cv = kfold , n_jobs=8, verbose=2)

#3.훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4.결과
results = model.score(x_test, y_test)
print('결과:',results)
print('걸린시간:',end - start)
