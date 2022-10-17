import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes,load_iris,load_wine,fetch_covtype,load_boston
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,f1_score

datasets = load_boston()
x, y =datasets.data, datasets.target

print(x.shape, y.shape)   #(178, 13) (178,)

x_train , x_test, y_train, y_test = train_test_split(
    x,y, random_state=123, train_size=0.8, shuffle=True)

Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

# Bootstrap Aggregation 
#Bagging 한가지 모델을 여러번 돌려서 사용 
from sklearn.ensemble import BaggingClassifier,BaggingRegressor #Bagging 앙상블 모델엣 가장많이 사용함 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,GradientBoostingRegressor

model1 = BaggingRegressor(DecisionTreeRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model2 = BaggingRegressor(RandomForestRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model3 = BaggingRegressor(GradientBoostingRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model4 = BaggingRegressor(XGBRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )


# model1 = DecisionTreeClassifier()
# model2 = RandomForestClassifier()
# model3 = GradientBoostingClassifier()
# model4 = XGBClassifier()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측
result1 = model1.score(x_test,y_test)
print("model1.score:",result1)

from sklearn.metrics import accuracy_score, r2_score

y_predict = model1.predict(x_test)
acc1 = r2_score(y_test,y_predict)

print( 'score1 :',acc1)
print(model1) 
print("===================================")

result2 = model2.score(x_test,y_test)
print("model2.score:",result2)


y_predict2 = model2.predict(x_test)
acc2 = r2_score(y_test,y_predict2)

print( 'score2 :',acc2)
print(model2) 
print("===================================")

result3 = model3.score(x_test,y_test)
print("model3.score3:",result3)


y_predict3 = model3.predict(x_test)
acc3 = r2_score(y_test,y_predict3)

print( 'score3 :',acc3)
print(model3)
print("===================================")

result4 = model4.score(x_test,y_test)
print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
acc4 = r2_score(y_test,y_predict4)

print( 'acc :',acc4)
print(model4) 
print("===================================")



# model1.score: 0.7782134337024393
# score1 : 0.7782134337024393
# BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# model2.score: 0.7780606806083736
# score2 : 0.7780606806083736
# BaggingRegressor(base_estimator=RandomForestRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# model3.score3: 0.8344599186091516
# score3 : 0.8344599186091516
# BaggingRegressor(base_estimator=GradientBoostingRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# model4.score: 0.8328611984546308
# acc : 0.8328611984546308
# BaggingRegressor(base_estimator=XGBRegressor(base_score=None, booster=None,