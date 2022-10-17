from unittest import result
import numpy as np
from sklearn.datasets import load_wine
from sympy import re


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

#x = np.delete(x, 1, axis=1)

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, shuffle=True, random_state=1234 
) 

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1] #features
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Impotances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)
    

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
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
# #     plt.subplot(2, 2, i+1)
# #     plot_feature_importances(models[i])
# #     if str(models[i]).startswith("XGB") : 
# #         plt.title('XGBRegressor')
# #     else :
# #         plt.title(models[i])

# # plt.show()


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
DecisionTreeClassifier 의 결과값 :  0.8888888888888888
model.feature_importances :  [0.         0.         0.         0.         0.         0.
 0.42646862 0.         0.         0.40024104 0.         0.02087679
 0.15241354]
====================================================================================================
====================================================================================================
RandomForestClassifier 의 결과값 :  0.9444444444444444
model.feature_importances :  [0.10941167 0.04280391 0.01415138 0.0346182  0.02186835 0.05996269
 0.17270178 0.00990546 0.02255326 0.17003178 0.06387508 0.14139775
 0.13671869]
====================================================================================================
====================================================================================================
GradientBoostingClassifier 의 결과값 :  0.8611111111111112
model.feature_importances :  [3.68676151e-03 3.92748014e-02 5.39752212e-03 5.41297429e-03
 6.69829417e-03 4.79113613e-06 7.61699220e-02 1.46300150e-03
 3.00606935e-04 3.08199734e-01 5.25349696e-05 2.61231268e-01
 2.92107787e-01]
====================================================================================================
====================================================================================================
XGBClassifier 의 결과값 :  0.8888888888888888
model.feature_importances :  [0.00715684 0.05944314 0.01588025 0.         0.03330867 0.00224883
 0.07457907 0.00876752 0.03545462 0.15266728 0.01350239 0.41068745
 0.18630388]
====================================================================================================
2. 컬럼 삭제 후 결과값
DecisionTreeClassifier 의 스코어:  0.9305555555555556
DecisionTreeClassifier 의 드랍후 스코어:  0.8888888888888888
RandomForestClassifier 의 스코어:  0.9722222222222222
RandomForestClassifier 의 드랍후 스코어:  0.9444444444444444
GradientBoostingClassifier 의 스코어:  0.9583333333333334
GradientBoostingClassifier 의 드랍후 스코어:  0.8611111111111112
XGB 의 스코어:  0.9583333333333334
XGB 의 드랍후 스코어:  0.9166666666666666
'''