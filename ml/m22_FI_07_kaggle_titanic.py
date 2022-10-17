# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 

from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import math
import numpy as np
import pandas as pd
#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
x = np.array(x)
x = np.delete(x,[4,6], axis=1)
y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측

from sklearn.metrics import accuracy_score, r2_score

result = model1.score(x_test,y_test)
print("model.score:",result)

y_predict = model1.predict(x_test)
acc = accuracy_score(y_test,y_predict)

print( 'accuracy_score :',acc)
print(model1,':',model1.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")




result2 = model2.score(x_test,y_test)
print("model2.score:",result2)

y_predict2 = model2.predict(x_test)
acc2 = accuracy_score(y_test,y_predict2)

print( 'accuracy2_score :',acc2)
print(model2,':',model2.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")




result3 = model3.score(x_test,y_test)
print("model3.score:",result3)

y_predict3 = model3.predict(x_test)
acc3 = accuracy_score(y_test,y_predict3)

print( 'accuracy3_score :',acc3)
print(model3,':',model3.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")



result4 = model4.score(x_test,y_test)
print("model4.score:",result4)

y_predict4 = model4.predict(x_test)
acc4 = accuracy_score(y_test,y_predict4)

print( 'accuracy4_score :',acc4)
print(model4,':',model4.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")
# 삭제후 
# model.score: 0.8268156424581006
# accuracy_score : 0.8268156424581006
# DecisionTreeClassifier() : [0.10301101 0.33110716 0.23788123 0.0495797  0.2784209 ]
# ===================================
# model2.score: 0.8603351955307262
# accuracy2_score : 0.8603351955307262
# RandomForestClassifier() : [0.08560709 0.28300849 0.27950585 0.04720047 0.3046781 ]
# ===================================
# model3.score: 0.8435754189944135
# accuracy3_score : 0.8435754189944135
# GradientBoostingClassifier() : [0.1436283  0.49550638 0.15074869 0.04155378 0.16856285]     
# ===================================
# model4.score: 0.8603351955307262
# accuracy4_score : 0.8603351955307262
# XGBClassifier


# 삭제전 
# model.score: 0.8212290502793296
# accuracy_score : 0.8212290502793296
# DecisionTreeClassifier() : [0.0993953  0.32988035 0.22231528 0.04055697 0.03727911 0.25811193 
#  0.01246107]
# ===================================
# model2.score: 0.8491620111731844
# accuracy2_score : 0.8491620111731844
# RandomForestClassifier() : [0.08686047 0.26535174 0.26206507 0.05051479 0.03871302 0.26611846 
#  0.03037646]
# ===================================
# model3.score: 0.8603351955307262
# accuracy3_score : 0.8603351955307262
# GradientBoostingClassifier() : [0.14059592 0.4890627  0.14394682 0.04110958 0.00358098 0.1653559
#  0.0163481 ]
# ===================================
# model4.score: 0.8603351955307262
# accuracy4_score : 0.8603351955307262
# XGBClassifier