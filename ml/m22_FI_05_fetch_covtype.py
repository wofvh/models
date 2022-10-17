# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import numpy as np
#1. 데이터
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']
# x = np.delete(x,[6,7,8,14,15,16], axis=1) 

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
print(model1,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")




result2 = model2.score(x_test,y_test)
print("model2.score:",result2)

y_predict2 = model2.predict(x_test)
acc2 = accuracy_score(y_test,y_predict2)

print( 'accuracy2_score :',acc2)
print(model2,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")




result3 = model3.score(x_test,y_test)
print("model3.score:",result3)

y_predict3 = model3.predict(x_test)
acc3 = accuracy_score(y_test,y_predict3)

print( 'accuracy3_score :',acc3)
print(model3,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")



result4 = model4.score(x_test,y_test)
print("model4.score:",result4)

y_predict4 = model4.predict(x_test)
acc4 = accuracy_score(y_test,y_predict4)

print( 'accuracy4_score :',acc4)
print(model4,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

# 삭제후 
# accuracy_score : 0.8194444444444444
# DecisionTreeClassifier() :
# ===================================
# model2.score: 0.9777777777777777
# accuracy2_score : 0.9777777777777777
# RandomForestClassifier() :
# ===================================
# model3.score: 0.9583333333333334
# accuracy3_score : 0.9583333333333334
# GradientBoostingClassifier() :
# ===================================
# model4.score: 0.9611111111111111
# accuracy4_score : 0.9611111111111111
# XGBClassifier

# 삭제전 
# model.score: 0.8305555555555556
# accuracy_score : 0.8305555555555556
# DecisionTreeClassifier() :
# ===================================
# model2.score: 0.975
# accuracy2_score : 0.975
# RandomForestClassifier() :
# ===================================
# model3.score: 0.9583333333333334
# accuracy3_score : 0.9583333333333334
# GradientBoostingClassifier() :
# ===================================
# model4.score: 0.9638888888888889
# accuracy4_score 

# 삭제후 

# model.score: 0.9666666666666667
# r2_score1 : 0.958100558659218
# DecisionTreeClassifier() : [0.02506789 0.06761888 0.90731323]
# ===================================
# model1.score: 0.9333333333333333
# r2_score2 : 0.9162011173184358
# RandomForestClassifier() : [0.21475798 0.37988318 0.40535883]
# ===================================
# model2.score3: 0.9666666666666667
# r2_score3 : 0.958100558659218
# GradientBoostingClassifier() : [0.01294319 0.64726702 0.33978979]
# ===================================
# model4.score: 0.9666666666666667
# r2_score4 : 0.958100558659218
# XGBClassifier : [0.01042643 0.8341722  0.15540144]
# ===================================


# 삭제전 
# DecisionTreeClassifier() : [0.03338202 0.         0.56740948 0.39920851]
# RandomForestClassifier() : [0.10385929 0.03867157 0.39319982 0.46426933]
# GradientBoostingClassifier() : [0.00482361 0.01545806 0.3617882  0.61793013]
# XGBClassifier : [0.00912187 0.0219429  0.678874   0.29006115]