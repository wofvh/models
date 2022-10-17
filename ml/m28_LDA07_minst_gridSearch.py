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
from keras.datasets import mnist
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold

import xgboost as xg
print('xgboost:', xg.__version__)


(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) # (70000, 784)
print(x.shape)#(70000, 784)
y= np.append(y_train, y_test) # (70000,)

print(x.shape)  #(178, 13)


le = LabelEncoder()
y = le.fit_transform(y)


# pca = PCA(n_components=20)   #54 >10
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x,y)
x = lda.transform(x)
print(x)

# pca_EVR = pca.explained_variance_ratio_

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

x_train , x_test, y_train, y_test = train_test_split(
    x,y, train_size= 0.8, random_state=1234 ,shuffle=True
)

print(np.unique(y_train, return_counts= True))

parameters = [
    {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
     'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
    {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
     'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
   
    ]  

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 

model = GridSearchCV(XGBClassifier(tree_method ='gpu_hist', predictor = 'gpu_predictor', gpu_id = 0,),parameters,verbose=1,
                     refit=True,n_jobs=-1) 


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
# 결과: 0.5711428571428572
# 걸린시간: 353.932089805603