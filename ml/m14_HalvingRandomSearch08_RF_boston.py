from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score,r2_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor #공부하자 

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = [
    {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
     'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
    {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
     'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
    ]     
#2. 모델 구성
model = HalvingRandomSearchCV(RandomForestRegressor(),parameters,cv=kfold,verbose=1,
                     refit=True,n_jobs=-1) 

#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train,y_train) 
end = time.time()- start

print("최적의 매개변수 :",model.best_estimator_)
print("최적의 파라미터 :",model.best_params_)
print("best_score :",model.best_score_)
print("model_score :",model.score(x_test,y_test))
y_predict = model.predict(x_test)
print('accuracy_score :',r2_score(y_test,y_predict))
y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',r2_score(y_test,y_predict))
print("걸린 시간 :",round(end,2),"초")
#==================gridsearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=3, n_estimators=200,
#                       n_jobs=-1)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 3,
# 'min_samples_split': 2, 'n_estimators': 200, 'n_jobs': -1}  
# best_score : 0.8100938812390837
# model_score : 0.8669310921614444
# accuracy_score : 0.8669310921614444
# 최적 튠  ACC : 0.8669310921614444
# 걸린 시간 : 19.21 초

#==================randomsearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=3, min_samples_split=3,
#                       n_estimators=200, n_jobs=-1)
# 최적의 파라미터 : {'n_jobs': -1, 'n_estimators': 200, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_depth': 8}  
# best_score : 0.803984689290913
# model_score : 0.8735859448125933
# accuracy_score : 0.8735859448125933
# 최적 튠  ACC : 0.873585944812593

#==================halvinggridsearch
# 최적의 매개변수 : RandomForestRegressor(max_depth=6, min_samples_leaf=3, n_estimators=200,
#                       n_jobs=2)
# 최적의 파라미터 : {'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 200, 'n_jobs': 2}   
# best_score : 0.8087518325656096
# model_score : 0.8699151320285012
# accuracy_score : 0.8699151320285012
# 최적 튠  ACC : 0.8699151320285012
# 걸린 시간 : 20.84 초
#==================halvingrandomsearch