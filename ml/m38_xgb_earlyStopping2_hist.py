from unittest import result
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRFRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings(action="ignore")


#1.데이터 
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569, 30) (569,)

x_train , x_test , y_train , y_test = train_test_split( x,y,
    shuffle=True, random_state=123 ,train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kflod = KFold(n_splits=5 , shuffle=True, random_state=123)


#2모델
model = XGBClassifier(random_state=123,
                      n_setimators=1000,
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,
                    )

# model = GridSearchCV(xgb, parameters, cv = kflod , n_jobs=8)

model.fit(x_train,y_train,
          early_stopping_rounds = 20, eval_set=[(x_train,y_train),(x_test,y_test)],
           #eval_set=[(x_test,y_test)],
           eval_metric ='error'
           #회기 : rmse,mae,rmsle...
           #이진 : error , auc... logloss..
           #다중 : merror, mlogloss...
          )

# best_params = model.best_params_
# print('최상의 매개변수:', model.best_param_)
# print('최상의 점수:', model.best_score_)

results =model.score(x_test,y_test)
print('최종 스코어:',results )

y_predict = model.predict(x_test)
acc= accuracy_score(y_test,y_predict)
print('진짜 최종TEST점수:', acc)

print('==========================================')

hist = model.evals_result()
print(hist)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.plot(hist["validation_0"]["error"])
plt.plot(hist["validation_1"]["error"])
plt.xlabel('round')
plt.ylabel('error')
plt.title('XGBoost')
plt.show()



plt.subplot(2,1,1)
plt.plot(hist["validation_0"]["error"])
plt.subplot(2,1,2)
plt.plot(hist["validation_1"]["error"])
plt.show()

#[실습]
#그래프 그려봐 

# n_estimators
# 최상의 매개변수: {'n_estimators': 100}
# 최상의 점수: 0.9692307692307691
# learning_rate'
# 최상의 매개변수: {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수: 0.9648351648351647
# max_depth':
# 최상의 매개변수: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수: 0.9692307692307691
# 'gamma':
# 최상의 매개변수: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수: 0.9692307692307691
# 'min_child_weight': 
# 최상의 매개변수: {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100}
# 최상의 점수: 0.9736263736263737
# subsample': 
# 최상의 매개변수: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.5}
# 최상의 점수: 0.9736263736263737
# 'colsample_bytree': 0.5
# 최상의 매개변수: {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수: 0.9736263736263737

# reg_alpha
# 최상의 매개변수: {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'subsample': 1}
# 최상의 점수: 0.9736263736263737

# # reg_alambda
# da': 0, 'subsample': 1}
# 최상의 점수: 0.9736263736263737
