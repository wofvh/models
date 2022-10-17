# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 칼럼들을 제거하여
# 데이터셋 재구성 후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

# 결과비교
# 1. DecisionTree
# 기존 acc: 
# 칼럼삭제 후 acc:

import numpy as np
from sklearn.datasets import load_iris
from sympy import re


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234 
) 

# import matplotlib.pyplot as plt

# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1] #features
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Impotances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model)
    

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,\
    RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier


models = [DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]


# for i in range(len(models)) :
#     model = models[i]
#     name = str(model).strip('()')
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
#     fimp = model.feature_importances_
#     print("="*100)
#     print(name,'의 결과값 : ', result)
#     print('model.feature_importances : ', fimp)
#     print("="*100)  
#     #plt.subplot(2, 2, i+1)
#     #plot_feature_importances(models[i])#
# #    if str(models[i]).startswith("XGB") : 
# #        plt.title('XGBRegressor')
# #    else :
# #        plt.title(models[i])

# #plt.show()


for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.8, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)
        
        

'''
1. 컬럼 삭제 하기 전 결과값
====================================================================================================
DecisionTreeClassifier 의 결과값 :  1.0
model.feature_importances :  [0.01669101 0.01669101 0.56740948 0.39920851]
====================================================================================================
====================================================================================================
RandomForestClassifier 의 결과값 :  1.0
model.feature_importances :  [0.08662243 0.03179796 0.41418795 0.46739166]
====================================================================================================
====================================================================================================
GradientBoostingClassifier 의 결과값 :  1.0
model.feature_importances :  [0.00649485 0.01290594 0.40522355 0.57537566]
====================================================================================================
====================================================================================================
XGBClassifier 의 결과값 :  1.0
model.feature_importances :  [0.00912187 0.0219429  0.678874   0.29006115]
====================================================================================================
2. 컬럼 삭제 후 결과값
====================================================================================================
DecisionTreeClassifier 의 결과값 :  1.0
model.feature_importances :  [0.01669101 0.07659085 0.90671814]
====================================================================================================
====================================================================================================
RandomForestClassifier 의 결과값 :  1.0
model.feature_importances :  [0.17862096 0.41075434 0.4106247 ]
====================================================================================================
====================================================================================================
GradientBoostingClassifier 의 결과값 :  1.0
model.feature_importances :  [0.01606579 0.37229802 0.6116362 ]
====================================================================================================
====================================================================================================
XGBClassifier 의 결과값 :  1.0
model.feature_importances :  [0.01531648 0.71701634 0.26766717]
====================================================================================================
DecisionTreeClassifier 의 스코어:  1.0
DecisionTreeClassifier 의 드랍후 스코어:  1.0
RandomForestClassifier 의 스코어:  1.0
RandomForestClassifier 의 드랍후 스코어:  1.0
GradientBoostingClassifier 의 스코어:  1.0
GradientBoostingClassifier 의 드랍후 스코어:  1.0
XGB 의 스코어:  1.0
XGB 의 드랍후 스코어:  1.0
'''