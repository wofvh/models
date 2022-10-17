from unittest import result
import numpy as np
import py
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes 
from sklearn import datasets
from sklearn.utils import all_estimators
#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test , y_train,y_test = train_test_split(x,y,
                                                    train_size = 0.8, shuffle=True,
                                                    random_state=72)
print(y_test.shape)


n_splits = 5 #다섯 번씩 모든 데이터를 훈련해준다고 지정 

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#Kfold 모든 데이터를 버리는거 없이 나누어줌 

print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)
# (89,)
# (89, 10)
# (353, 10)
# (89,)
# (353,)

#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC()

#컴파일 훈련
scores = cross_val_score(model , x_train, y_train, cv=kfold)

print('ACC :', scores,'\n cross_val_score:', round(np.mean(scores),4))
y_predict = cross_val_score(model,x_test,y_test,cv=kfold)
print(y_predict)
acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC',acc)
