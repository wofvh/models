from tracemalloc import start
from turtle import shape
from unittest import result
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,fetch_covtype,load_digits
from sklearn.datasets import load_breast_cancer,load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xg
print('xgboost:', xg.__version__)

datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape)  #(150, 4)  
print(y.shape)

le = LabelEncoder()
y = le.fit_transform(y)
print(y.shape)
print(np.unique(y,return_counts= True))
#(array([0, 1, 2], dtype=int64), array([37, 44, 39], dtype=int64))
# pca = PCA(n_components=20)   #54 >10
# x = pca.fit_transform(x)



lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
x = lda.transform(x)
print(x.shape)
print(y.shape)

print(np.unique(y, return_counts= True))

# pca_EVR = pca.explained_variance_ratio_

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

x_train , x_test, y_train, y_test = train_test_split(
    x,y, train_size= 0.8, random_state=123 ,shuffle=True
)

print(np.unique(y_train, return_counts= True))
print(x_train.shape)
'''
#(array([0, 1, 2], dtype=int64), array([37, 44, 39], dtype=int64))
#       dtype=int64))


#2. 모델
from xgboost import XGBClassifier,XGBRegressor
model = XGBClassifier(tree_method ='gpu_hist', predictor = 'gpu_predictor', gpu_id = 0,)


#3.훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4.결과
results = model.score(x_test, y_test)
print('결과:',results)
print('걸린시간:',end - start)


# LDA
# 결과: 0.9666666666666667
# 걸린시간: 0.5678057670593262
'''