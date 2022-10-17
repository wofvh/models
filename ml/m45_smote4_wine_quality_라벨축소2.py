#csv로 만들어 
from selectors import SelectSelector
from tabnanny import verbose
from sklearn.datasets import load_breast_cancer, load_wine,fetch_covtype
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score,f1_score
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE 

path = 'D:/study_data/_data/'
datasets = pd.read_csv(path + 'winequality-white.csv',
                       index_col=None, header=0, sep=';') # sep';'< csv 

print(datasets.shape)   #(4898, 12)
print(datasets.describe())   #(4898, 12)
print(datasets.info())

# x = datasets.iloc[:, :-1]   #x값 슬라이스로 마지막 puality (컬럼)부분을 제외
# y = datasets.iloc[:, -1]    #y값 슬라이스로 마지막 puality (컬럼)부분을 y 값으로 지정

datasets2 = datasets.to_numpy()    #pandas 를 numpy로 바꿀때 

print(type(datasets2))
print(datasets.shape)

x = datasets2[:, :11]
y = datasets2[:, 11]
print(x.shape, y.shape)

print(np.unique(y, return_counts = True)) #다중분류에서는 데이터 무조건 확인하기
# (array([3., 4., 5., 6., 7., 8., 9.]), array ([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(datasets['quality'].value_counts())

print(y[:20])
#[6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]
       


for index, value in enumerate(y):
    if value == 9 :     #9.0   5
        y[index] = 7 
    elif  value  ==8:
        y[index] = 7
    elif  value  ==7:
        y[index] = 7
    elif  value  ==6:
        y[index] = 6
    elif  value  ==5:
        y[index] = 5
    elif  value  ==4:
        y[index] = 4
    elif  value  ==3:
        y[index] = 4
    else:
        y[index] = 0


print(np.unique(y, return_counts = True))
# (array([4., 5., 6., 7.]), array([ 183, 1457, 2198, 1060], dtype=int64))


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123, stratify= y
) 

smote = SMOTE(random_state=123, k_neighbors = 3)                             #SMOTE 증폭 
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts = True))


#2, 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3.훈련
model.fit(x_train,y_train)

#4. 평가,예측
y_predict  = model.predict(x_test)

score = model.score(x_test,y_test)  
print("model.score:", score)
print('acc_score:',accuracy_score (y_test,y_predict))
print('f1_score(macro): ',f1_score(y_test,y_predict , average="macro")) #F1_score 은 2진분류에서 많이 사용 다중분류에서 average="macro" 사용
print('f1_score(micro): ',f1_score(y_test,y_predict , average="micro")) #F1_score 은 2진분류에서 많이 사용 
########과제 프리시즌 리콜 precis


########과제 프리시즌 리콜 이해해서 보내기

#아웃 레이어 추가 안 했을떄
# Name: quality, dtype: int64
# model.score: 0.7255102040816327
# acc_score: 0.7255102040816327
# f1_score(macro):  0.4415268598572438
# f1_score(micro):  0.7255102040816328

#아웃레이어로 했을때 
# model.score: 0.7489361702127659
# acc_score: 0.7489361702127659
# f1_score(macro):  0.6663772634873866
# f1_score(micro):  0.7489361702127659
