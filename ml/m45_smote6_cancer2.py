#1 3 5 7
#0 2 1 2

# 라벨 0을 112개 삭제해서 재구성 

#smote 넣어서 만들기 
#넣은거 안 넣은거 
# smote 넣기 
# 비교

from selectors import SelectSelector
from sklearn.datasets import load_breast_cancer , load_diabetes , load_iris ,fetch_california_housing,load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE 


#1.데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569, 30) (569,)

# 라벨 축소하기
for i in range(len(y)):
    if y[i] < 4:
        y[i] = 4 
    elif y[i] > 8:
        y[i] = 8
print(np.unique(y))  # [4, 5, 6, 7, 8]

print(np.unique(y, return_counts = True))
# (array([4]), array([569], dtype=int64))


'''
# for index, value in enumerate(y):
#     if value == 9 :
#         y[index] = 2
#     elif value == 8 :
#         y[index] = 2
#     elif value ==  7:
#         y[index] = 2
#     elif value == 6 :
#         y[index] = 1
#     elif value == 5 :
#         y[index] = 0
#     elif value == 4 :
#         y[index] = 0
#     elif value == 3 :
#         y[index] = 0
#     else : 
#         y[index] = 10


x_train , x_test , y_train , y_test = train_test_split( x,y,
    shuffle=True, random_state=123 ,train_size=0.8
)

print(np.unique(y_train, return_counts = True))

smote = SMOTE(random_state=123, k_neighbors = 3)                             #SMOTE 증폭 
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts = True))


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kflod = KFold(n_splits=5 , shuffle=True, random_state=123)

#2모델
model = XGBClassifier(random_state=123,
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1
                    )

model.fit(x_train,y_train,
          early_stopping_rounds = 200, eval_set=[(x_train,y_train),(x_test,y_test)],
           #eval_set=[(x_test,y_test)],
           eval_metric ='logloss')

results =model.score(x_test, y_test)
print('최종 점수:',results )

y_predict = model.predict(x_test)
acc= accuracy_score(y_test, y_predict)
print("진짜 최종TEST점수:", acc)

print(model.feature_importances_)
'''