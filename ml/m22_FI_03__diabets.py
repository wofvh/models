from unittest import result
import numpy as np
from sklearn.datasets import load_diabetes
from sympy import re


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

allfeature = round(x.shape[1]*0.2, 1)
print('자를 갯수: ', int(allfeature))


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234 
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


models = [DecisionTreeRegressor(),RandomForestRegressor(),GradientBoostingRegressor(),XGBRegressor()]


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
        print(featurelist)
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
DecisionTreeRegressor 의 결과값 :  -0.1068881848170371
model.feature_importances :  [0.07225872 0.01471705 0.34536476 0.08939058 0.02359742 0.09628121
 0.05902221 0.01510055 0.16298507 0.12128243]
====================================================================================================
====================================================================================================
RandomForestRegressor 의 결과값 :  0.4393646163332814
model.feature_importances :  [0.05842269 0.01291122 0.32343341 0.08308942 0.04490954 0.06238879
 0.05826202 0.03119728 0.23983004 0.08555558]
====================================================================================================
====================================================================================================
GradientBoostingRegressor 의 결과값 :  0.4218927960226726
model.feature_importances :  [0.04581274 0.01646282 0.33588192 0.09557779 0.03115747 0.06720554
 0.03842033 0.01440577 0.27643673 0.07863891]
====================================================================================================
====================================================================================================
XGBRegressor 의 결과값 :  0.26078151031491137
model.feature_importances :  [0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191
 0.06551369 0.17944618 0.13779876 0.08540721]
====================================================================================================
2. 컬럼 삭제 후 결과값
DecisionTreeRegressor 의 스코어:  -0.11436789943081438
DecisionTreeRegressor 의 드랍후 스코어:  -0.04972498333301423
RandomForestRegressor 의 스코어:  0.4223129983063971
RandomForestRegressor 의 드랍후 스코어:  0.4251750421097895
GradientBoostingRegressor 의 스코어:  0.4165006725618192
GradientBoostingRegressor 의 드랍후 스코어:  0.406906613340267
XGB 의 스코어:  0.26078151031491137
XGB 의 드랍후 스코어:  0.3080091954400702
'''