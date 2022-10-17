from dataclasses import dataclass
from tabnanny import verbose
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold, cross_val_score

#데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split,StratifiedKFold

x_train ,x_test, y_train ,y_test = train_test_split(x,y, train_size=0.8, 
                                                   shuffle=True,random_state=1234 )

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

Parameters= [
    {"RF__n_estimators":[100,200], "RF__max_depth":[6,10,12],'RF__min_samples_leaf':[3, 10]},
    {"RF__max_depth": [6, 18, 10 ,12],"RF__min_samples_leaf" :[13, 15, 7 ,10],},
    {'RF__min_samples_leaf':[13,5,7,10],"RF__n_jobs":[14,20,15,12],"RF__max_depth":[6, 8, 10 ,12]},
    {"RF__min_samples_split":[2,13,5,10],"RF__min_samples_split":[15,20,15,12],'RF__min_samples_leaf':[12,25,7,10],},
    {'RF__n_jobs':[-1,2,4],"RF__max_depth":[6,10,12]}
] 


# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


#모델구성

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline ,Pipeline

# model= SVC()
# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())

pipe = Pipeline([("minmax", MinMaxScaler()),("RF",RandomForestClassifier())],verbose=1)


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV,HalvingRandomSearchCV
GridSearchCV
model = GridSearchCV(pipe,Parameters,cv=kfold,verbose=1)
# model = GridSearchCV(pipe, parameters,cv=5,verbose=1)


#훈련
model.fit(x_train,y_train)

#평가예측
result = model.score(x_test,y_test)

print("model.score:", result)


# model.score: 1.0

# [Pipeline] ................ (step 2 of 2) Processing RF, total=   0.1s
# model.score: 0.4604296264610215