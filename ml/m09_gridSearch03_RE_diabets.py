import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn import datasets                 #분류 & 회기
from sklearn.linear_model import LogisticRegression,LinearRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor    #
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor     # 
from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor  # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold
from sklearn.model_selection import KFold, cross_val_score , GridSearchCV
#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=77)

print(y_train.shape)     #(353,)
print(x_train.shape)     #(353, 10)



#2. 모델구성\
# model = LinearRegression()      #  결과 r2: 0.7395013831787173
# model = KNeighborsRegressor()      # 결과 r2: 0.7692717948717949

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

Parameters= [
    {"n_estimators":[100,200], "max_depth":[6,10,12],'min_samples_leaf':[3, 10]},
    {"max_depth": [6, 8, 10 ,12],"min_samples_leaf" :[3, 5, 7 ,10],'min_samples_leaf':[3, 10]},
    {'min_samples_leaf':[3,5,7,10],"n_jobs":[14,20,15,12],"max_depth":[6, 8, 10 ,12],"max_depth": [6, 8, 10 ,12]},
    {"min_samples_split":[2,3,5,10],"min_samples_leaf" :[3, 5, 7 ,10],'min_samples_leaf':[12,25,7,10],},
    {'n_jobs':[-1,2,4],"min_samples_split":[15,20,15,12],"max_depth":[6,10,12]}
]

model = GridSearchCV(RandomForestRegressor(),Parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)  #n_jobs cpu 갯수사용 정의 예) 1 cup 1 -1 cup8 

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


# Fitting 5 folds for each of 136 candidates, totalling 680 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=7, min_samples_split=5)
# 최적의 파라미터 :  {'min_samples_leaf': 7, 'min_samples_split': 5}
# best_score_: 0.4444742936046834
# model.score: 0.4866901903274303
# r2_score: 0.4866901903274303
# 최적 튠 ACC: 0.4866901903274303
