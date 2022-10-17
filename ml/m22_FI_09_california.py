from unittest import result
import numpy as np
from sklearn.datasets import fetch_california_housing
from sympy import re


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

#x = np.delete(x, 1, axis=1)

allfeature = round(x.shape[1]*0.2, 0)
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

# plt.show()
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
DecisionTreeRegressor 의 결과값 :  0.6353541055864678
model.feature_importances :  [0.52245156 0.05510963 0.03224017 0.03130079 0.12890089 0.11300856
 0.11698839]
====================================================================================================
====================================================================================================
RandomForestRegressor 의 결과값 :  0.8056162933146739
model.feature_importances :  [0.52400013 0.05054662 0.02930346 0.03300944 0.13471606 0.11088343
 0.11754086]
====================================================================================================
====================================================================================================
GradientBoostingRegressor 의 결과값 :  0.7766059047382794
model.feature_importances :  [0.60571664 0.01868548 0.00476961 0.00233388 0.12852339 0.11304196
 0.12692904]
====================================================================================================
====================================================================================================
XGBRegressor 의 결과값 :  0.823045985237328
model.feature_importances :  [0.47640604 0.04571545 0.02496676 0.02580887 0.16017456 0.12103622
 0.1458921 ]
====================================================================================================
2. 컬럼 삭제 후 결과값
DecisionTreeRegressor 의 스코어:  0.5999653477979593
DecisionTreeRegressor 의 드랍후 스코어:  0.6118782624297641
RandomForestRegressor 의 스코어:  0.8038905852214953
RandomForestRegressor 의 드랍후 스코어:  0.8071457307092794
GradientBoostingRegressor 의 스코어:  0.7868072463466687
GradientBoostingRegressor 의 드랍후 스코어:  0.7833602753783532
XGB 의 스코어:  0.8266362514646761
XGB 의 드랍후 스코어:  0.8307509185446069
'''

