from unittest import result
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score , GridSearchCV,HalvingGridSearchCV


#1.데이터
datasets = load_iris()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, shuffle= True,
                                                    random_state=72 )

print(y_test)

n_splits = 5
Kfold = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=66)


# #2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier    #
from sklearn.tree import DecisionTreeClassifier       # 
from sklearn.ensemble import RandomForestClassifier   # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 

Parameters= [
    {"n_estimators":[100,200], "max_depth":[6,12,14],'min_samples_leaf':[3, 10]},
    {"max_depth": [6, 82, 10 ,12],"min_samples_leaf" :[4, 16, 8,10],},
    {'min_samples_leaf':[3,15,71,10],"n_jobs":[14,20,9,12],"max_depth":[6, 18, 10 ,12]},
    {"min_samples_split":[2,3,15,10],"min_samples_split":[15,20,15,12],'min_samples_leaf':[12,25,7,10],},
    {'n_jobs':[-1,2,4],"max_depth":[6,11,12],"max_depth":[6, 8, 10 ,12]}
]

model = HalvingGridSearchCV(RandomForestClassifier(),Parameters, cv =Kfold, verbose=1, refit=True, n_jobs=-1)
#컴파일 훈련
import time
start = time.time()

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ",model.best_params_)

print('best_score_:', model.best_score_)

print("model.score:", model.score(x_test,y_test))

#평가,예측\
end_time = time.time()
  
y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))
# print('accuracy', results[1])

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test, y_pred_best))


# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=10, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 10, 'n_estimators': 200}
# best_score_: 0.9583333333333334
# model.score: 1.0
# accuracy_score: 1.0
# 최적 튠 ACC: 1.0