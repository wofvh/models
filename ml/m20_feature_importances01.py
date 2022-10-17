import numpy as np
from sklearn.datasets import load_iris

# 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=True, random_state=1234)


#2. 모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()

model = XGBClassifier()

#3. 훈련

model.fit(x_train,y_train)

#4.평가예측
result = model.score(x_test, y_test)
print("model.score:",result)

from sklearn.metrics import accuracy_score
y_predict =model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print('accuracy_score:', acc)

print("===================================")
print(model,':',model.feature_importances_)


# DecisionTreeClassifier() : [0.         0.01669101 0.07659085 0.90671814]
# GradientBoostingClassifier() : [0.0090424  0.01098409 0.2769728  0.70300071]
# RandomForestClassifier() : [0.08730228 0.0265098  0.44368654 0.44250138]
#  XGBClassifier [0.00912187 0.0219429  0.678874   0.29006115]