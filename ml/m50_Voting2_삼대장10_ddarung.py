import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.experimental import enable_iterative_imputer # IterativeImputer  입력시 사용하는 모듈 추가
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer 
warnings.filterwarnings(action="ignore")

# parameters_xgb = [
#     {'XGB__n_estimators' : [100,200,300,400,500],
#     'XGB__learning_rate' : [0.01,0.05,0.1,0.15],
#     'XGB__max_depth' : [3,5,7,10,15],
#     'XGB__gamma' : [0,1,2,3],
#     'XGB__colsample_bytree' : [0.8,0.9]}]

# parameters_rfr = [{
#     'RFR__bootstrap': [True], 'RFR__max_depth': [5, 10, None], 
#     'RFR__max_features': ['auto', 'log2'], 'RFR__n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}]

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (715, 9)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 knn 임퓨터 ####
imputer1 = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors default값은 3
imputer2 = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors default값은 3

# imputer1 = IterativeImputer(missing_values=np.nan, max_iter=10, tol=0.001)
# imputer2 = IterativeImputer(missing_values=np.nan, max_iter=10, tol=0.001)

print(train_set.isnull().sum())
imputer1.fit(train_set) # 데이터프레임에 적용하기 위해 fit()함수 사용
imputer2.fit(test_set) # 데이터프레임에 적용하기 위해 fit()함수 사용
train_set_imputer = imputer1.transform(train_set) # transform()함수를 사용하여 데이터프레임을 적용하기 위해 transform()함수 사용
test_set_imputer = imputer2.transform(test_set) # transform()함수를 사용하여 데이터프레임을 적용하기 위해 transform()함수 사용
print(train_set.shape) # (1459, 10)
train_set = pd.DataFrame(train_set_imputer, columns=train_set.columns) # 데이터프레임을 데이터프레임으로 변환하기 위해 DataFrame()함수 사용
test_set = pd.DataFrame(test_set_imputer, columns=test_set.columns) # 데이터프레임을 데이터프레임으로 변환하기 위해 DataFrame()함수 사용
############################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count'] 
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=123) # train_size는 학습데이터의 비율, random_state는 난수 시드
                                                    

Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델
from sklearn.ensemble import VotingClassifier,VotingRegressor
from xgboost import XGBClassifier,XGBRFRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
xg = XGBRFRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor() #(verbose=False)#catboost vervose가 많음 ! 그래서 다른모델이랑 성능비교 시에는 주석처리

#voting 은 hard &soft가있음 #estimators= 두개이상은 리스트로 넣어줘야함
model = VotingRegressor(estimators=[('xg', xg), ('cat', cat),("lg", lg)], ) 

#3. 평가예측
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)
print(model.score(x_test,y_test))

score = r2_score(y_test,y_predict)
print("보팅결과 : ", round(score,4 ))

# 보팅결과 :  0.9912

classifier = [cat,xg, lg,]

for model in classifier:  #model2는 모델이름 # 
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    score2 = r2_score(y_test,y_predict)
    class_name = model.__class__.__name__  #<모델이름 반환해줌 
    print("{0}정확도 : {1:.4f}".format(class_name, score2)) # f = format
    
print("보팅결과 : ", round(score,4 ))
    
# CatBoostRegressor정확도 : 0.8085
# XGBRFRegressor정확도 : 0.7509
# LGBMRegressor정확도 : 0.7667
# 보팅결과 :  0.7912