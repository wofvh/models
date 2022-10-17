from unittest import result
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import true
from tensorboard import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold


import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#                                                     train_size=0.8, shuffle= True,
#                                                     random_state=66 )

n_splits = 5
# Kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)
Kfold = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=66)


# #2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier    #
from sklearn.tree import DecisionTreeClassifier       # 
from sklearn.ensemble import RandomForestClassifier   # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 

model = SVC()

#컴파일 훈련
# from sklearn.model_selection

# model.fit(x_train, y_train)
scores = cross_val_score(model, x, y, cv=Kfold)

print('ACC :', scores,'\n cross_val_score:', round(np.mean(scores),4)) # cross_val_score: 0.9667
#평가,예측
# results = model.evaluate(x_test, y_test)
# print('loss:',results[0])
# print('accuracy', results[1])


# results = model.score(x_test,y_test)
# print(results)