import numpy as np
from sklearn.datasets import load_diabetes

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor # pip install xgboost
import matplotlib.pyplot as plt

def plot_feature_importances(model): # 그림 함수 정의
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
                # x                     y
    plt.yticks(np.arange(n_features), datasets.feature_names) # 눈금 설정
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features) # ylimit : 축의 한계치 설정

models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]
print(str(models[3]))

# 3. 훈련
plt.figure(figsize=(16,14))
for i in range(len(models)):
    models[i].fit(x_train, y_train)
    plt.subplot(2,2, i+1)
    plot_feature_importances(models[i])
    if str(models[i]).startswith('XGBRegressor'):
        plt.title('XGB()')
    else:
        plt.title(models[i])

plt.show()
# DecisionTreeClassifier() : [0.         0.01669101 0.07659085 0.90671814]
# GradientBoostingClassifier() : [0.0090424  0.01098409 0.2769728  0.70300071]
# RandomForestClassifier() : [0.08730228 0.0265098  0.44368654 0.44250138]
#  XGBClassifier [0.00912187 0.0219429  0.678874   0.29006115]
