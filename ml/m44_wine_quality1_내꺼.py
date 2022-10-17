#csv로 만들어 
from selectors import SelectSelector
from tabnanny import verbose
from sklearn.datasets import load_breast_cancer, load_wine,fetch_covtype
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

path = 'D:/study_data/_data/'
datasets = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';') # sep';'< csv 
x = datasets.iloc[:, :-1]   #x값 슬라이스로 마지막 puality (컬럼)부분을 제외
y = datasets.iloc[:, -1]    #y값 슬라이스로 마지막 puality (컬럼)부분을 y 값으로 지정

# print(x.shape)
# print(y.shape)

le = LabelEncoder()
y = le.fit_transform(y)

lda = LinearDiscriminantAnalysis(n_components=5)
lda.fit(x,y)
x = lda.transform(x)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, shuffle=True, random_state=1234 
) 

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

print(x_train.shape)    #(2938, 1)
print(y_train.shape)    #(2938,)
print(x_test.shape)     #(1960, 1)
print(y_test.shape)     #(1960,)



# parameters = {"n_estimators" : [100,],
#               "learning_rate" : [0.1],
#               'max_depth' : [3],
#               'gamma' :[1],
#               'reg_alpha':[0,],
#               'reg_lambda':[1],
#               } 


#2모델
model = XGBClassifier(random_state=123,
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,
                      tree_method ='gpu_hist', predictor = 'gpu_predictor', gpu_id = 0,
                    )




model.fit(x_train,y_train,
          early_stopping_rounds = 200, eval_set=[(x_train,y_train),(x_test,y_test)],
           #eval_set=[(x_test,y_test)],
           eval_metric = "mlogloss",verbose=2)


results =model.score(x_test, y_test)
print('최종 점수:',results )

y_predict = model.predict(x_test)
acc= accuracy_score(y_test, y_predict)
print("accuracy:", acc)

print(model.feature_importances_)

thresholds = model.feature_importances_
print("===============================")
# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape,select_x_test.shape)
    
#     selection_model= XGBClassifier(random_state=123,
#                       n_estimators=100,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gamma=1,)
#     selection_model.fit(select_x_train,y_train)
    
#     y_predict = selection_model.predict(select_x_test)
#     score = accuracy_score(y_test, y_predict)
    
#     print("thresh=%.3F, N=%d, accuracy_score:%.2f%%"
#           %(thresh, select_x_train.shape[1],score*100))