from xml.sax.handler import feature_external_ges
import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd

# 데이터
datasets = load_diabetes()
print(datasets['feature_names'])
x = datasets.data
y = datasets.target

x = pd.DataFrame (x, columns=[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']])

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=True, random_state=1234)


#2. 모델구성 
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

# model = DecisionTreeRegressor()
# model = GradientBoostingRegressor()
# model = RandomForestRegressor()
model = XGBRegressor()

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

import matplotlib.pyplot as plt

# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel('feature importances')
#     plt.ylabel('features')
#     plt.ylim(-1, n_features)
#     plt.title(model)
# plot_feature_importances(model)
# plt.show()

from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()