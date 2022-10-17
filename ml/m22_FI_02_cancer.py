# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)



# x = np.array(x)
# y = np.array(y) 

# x = np.delete(x,[10,11,12,13,14,15,16,17,18,19,20,21,23,29], axis=1) 
# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


print(x.shape,y.shape)
print(datasets.feature_names)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


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
# model.score: 0.9649122807017544
# accuracy_score : 0.9649122807017544
# DecisionTreeClassifier() 
# ===================================
# model2.score: 0.9736842105263158
# accuracy2_score : 0.9736842105263158
# RandomForestClassifier() 
# ===================================
# model3.score: 0.9736842105263158
# accuracy3_score : 0.9736842105263158
# GradientBoostingClassifier() 
# ===================================
# model4.score: 0.9736842105263158
# accuracy4_score : 0.9736842105263158
# XGBClassifier 

# 삭제전 
# model.score: 0.956140350877193
# accuracy_score : 0.956140350877193
# DecisionTreeClassifier() :
# ===================================
# model2.score: 0.9912280701754386
# accuracy2_score : 0.9912280701754386
# RandomForestClassifier() :
# ===================================
# model3.score: 0.9736842105263158
# accuracy3_score : 0.9736842105263158
# GradientBoostingClassifier() :
# ===================================
# model4.score: 0.9649122807017544
# accuracy4_score : 0.9649122807017544
# XGBClassifier
