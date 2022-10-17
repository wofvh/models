import numpy as np
from sklearn.datasets import load_diabetes

# 데이터
datasets = load_diabetes()
# Sam = Sam.drop(['일자'], axis=1)
print(datasets.feature_names)
print(datasets['feature_names'])


x = datasets.data
y = datasets.target

print(x.shape)

x = np.delete(x,[1], axis=1)   # (axis< 열) 
  

print(x.shape)


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


# DecisionTreeClassifier() : [0.         0.01669101 0.07659085 0.90671814]
# GradientBoostingClassifier() : [0.0090424  0.01098409 0.2769728  0.70300071]
# RandomForestClassifier() : [0.08730228 0.0265098  0.44368654 0.44250138]
#  XGBClassifier [0.00912187 0.0219429  0.678874   0.29006115]


# ===================================
# DecisionTreeRegressor() : [0.08095086 0.35345988 0.10244602 0.04177469 0.09009914 0.04074602
#  0.0210978  0.15372839 0.11569719]

# GradientBoostingRegressor() : [0.0455695  0.34259213 0.08639873 0.03761692 0.07001003 0.03859892
#  0.01424612 0.28055964 0.08440801]


# RandomForestRegressor() : [0.06195658 0.3177018  0.08132132 0.05209201 0.05856314 0.06241516
#  0.02901043 0.24611925 0.09082029]

# XGBRegressor() [0.0260045  0.29435775 0.05291139 0.05066145 0.06820749 0.07702424
#  0.20247848 0.14291398 0.08544067

# model.score: 0.39816887360815867
# r2_score: <function r2_score at 0x0000027A1748F5E0>
# ===================================
# RandomForestRegressor() : [0.05626905 0.3197349  0.0855404  0.0534594  0.0582387  0.05949721
#  0.03157539 0.24639448 0.08929046]
# RandomForestRegressor()

