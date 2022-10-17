from xml.sax.handler import feature_external_ges
import numpy as np
from sklearn.datasets import load_diabetes

# 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=True, random_state=1234)


#2. 모델구성 
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

# model = DecisionTreeRegressor()
# model = GradientBoostingRegressor()
model = RandomForestRegressor()
# model = XGBRegressor()

#3. 훈련

model.fit(x_train,y_train)

#4.평가예측
result = model.score(x_test, y_test)
print("model.score:",result)

from sklearn.metrics import accuracy_score ,r2_score
y_predict =model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score:', r2_score)

print("===================================")
print(model,':',model.feature_importances_)
print(model)

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('feature importances')
    plt.ylabel('features')
    plt.ylim(-1, n_features)
    plt.title(model)
plot_feature_importances(model)
plt.show()






# DecisionTreeClassifier() : [0.         0.01669101 0.07659085 0.90671814]
# GradientBoostingClassifier() : [0.0090424  0.01098409 0.2769728  0.70300071]
# RandomForestClassifier() : [0.08730228 0.0265098  0.44368654 0.44250138]
#  XGBClassifier [0.00912187 0.0219429  0.678874   0.29006115]