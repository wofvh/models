from mmap import ACCESS_WRITE
from sre_parse import FLAGS
import numpy as np
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from tensorboard import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.model_selection import KFold, cross_val_score , GridSearchCV
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold

# 1.데이터
datasets = load_boston()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.2, shuffle= True,
                                                    random_state=1234 )
n_splits = 5 #다섯 번씩 모든 데이터를 훈련해준다고 지정 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

Parameters= [
    {"n_estimators":[100,200], "max_depth":[6,10,12],'min_samples_leaf':[3, 10]},
    {"max_depth": [6, 8, 10 ,12],"min_samples_leaf" :[3, 5, 7 ,10],},
    {'min_samples_leaf':[3,5,7,10],"n_jobs":[14,20,15,12],"max_depth":[6, 8, 10 ,12]},
    {"min_samples_split":[2,3,5,10],"min_samples_split":[15,20,15,12],'min_samples_leaf':[12,25,7,10],},
    {'n_jobs':[-1,2,4],"max_depth":[6,10,12]}
]

# #2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression  #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier    
from sklearn.tree import DecisionTreeClassifier       
from sklearn.ensemble import RandomForestRegressor  #  RandomForestRegressor 회기
from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier 분류  

# model = SVC(C=1, kernel="linear", degree=3)
model = GridSearchCV(RandomForestRegressor(),Parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)  #n_jobs cpu 갯수사용 정의 예) 1 cup 1 -1 cup8 

# RandomForestClassifier#분류모델
#컴파일 훈련
model.fit(x_train, y_train)
print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ",model.best_params_)

print('best_score_:', model.best_score_)

print("model.score:", model.score(x_test,y_test))

#평가,예측\
  
y_predict = model.predict(x_test)
print('r2_score:', r2_score(y_test, y_predict))
# print('accuracy', results[1])

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC:', r2_score(y_test, y_pred_best))


# (tf282gpu) C:\study>C:/anaconda3/envs/tf282gpu/python.exe c:/study/ml/m09_gridSearch8_RE_boston.py
# Fitting 5 folds for each of 117 candidates, totalling 585 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_jobs=4)
# 최적의 파라미터 :  {'max_depth': 10, 'n_jobs': 4}
# best_score_: 0.7588883230679637
# model.score: 0.8385099246592218
# r2_score: 0.8385099246592218
# 최적 튠 ACC: 0.8385099246592218
