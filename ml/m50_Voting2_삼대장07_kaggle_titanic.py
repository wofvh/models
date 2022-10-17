import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier,VotingRegressor
from xgboost import XGBClassifier,XGBRFRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.experimental import enable_iterative_imputer # 이터러블 입력시 사용하는 모듈 추가
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer 

import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")

parameters = {"n_estimators" : [100,200,300,400,500,1000],
              "learning_rate" : [0.1, 0.2, 0,3, 0.5, 0.01, 0.001],
              'max_depth' : [None,2,3,4,5,6,7,8,9,10],
              'gamma' :[0, 1, 2, 3, 4, 5, 7, 10, 100],
            #   'min_child_weight':[0, 0.01, 0.1, 0.5, 1, 5, 10,100],
            #   'subsample': [0, 0.1, 0.2, 0.3,0.5, 0.7, 1],
            #   'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1],
            #   'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1],
            #   'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1],
            #   'reg_alpha':[0, 0.1, 0.01, 0.001, 1, 2, 10],
              'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10],
              } 


kfold = KFold(n_splits=5,shuffle=True,random_state=100)

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (418, 10)
# print(test_set.describe())

print(train_set.Pclass.value_counts())

Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")

sns.barplot(x="SibSp", y="Survived", data=train_set)


# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)



# print(test_set.columns)
# print(train_set.info()) # info 정보출력
# print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 1. 제거 ####

train_set = train_set.fillna({"Embarked": "C"})
# train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

# test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
# test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

#### 결측치 처리 knn 임퓨터 ####
imputer = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors default값은 3

# imputer1 = IterativeImputer(missing_values=np.nan, max_iter=10, tol=0.001)
# imputer2 = IterativeImputer(missing_values=np.nan, max_iter=10, tol=0.001)

train_set.Age = imputer.fit_transform(train_set.Age.values.reshape(-1,1)) # reshape(-1,1) : 1차원으로 만듬
test_set.Age = imputer.fit_transform(test_set.Age.values.reshape(-1,1)) # reshape(-1,1) : 1차원으로 만듬
test_set.Fare = imputer.fit_transform(test_set.Fare.values.reshape(-1,1)) # reshape(-1,1) : 1차원으로 만듬


print(train_set, test_set, train_set.shape, test_set.shape)

############################


x = train_set.drop(['Survived'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived'] 
print(y)
print(y.shape) # (891,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier() #(verbose=False)#catboost vervose가 많음 ! 그래서 다른모델이랑 성능비교 시에는 주석처리

#voting 은 hard &soft가있음 #estimators= 두개이상은 리스트로 넣어줘야함
model = VotingClassifier(estimators=[('xg', xg), ('cat', cat),("lg", lg)], voting='soft') 

#3. 평가예측
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)
print(model.score(x_test,y_test))

score = accuracy_score(y_test,y_predict)
print("보팅결과 : ", round(score,4 ))

# 보팅결과 :  0.9912

classifier = [cat,xg, lg,]

for model in classifier:  #model2는 모델이름 # 
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    score2 = accuracy_score(y_test,y_predict)
    class_name = model.__class__.__name__  #<모델이름 반환해줌 
    print("{0}정확도 : {1:.4f}".format(class_name, score2)) # f = format
    
print("보팅결과 : ", round(score,4 ))
    
# CatBoostClassifier정확도 : 0.8492
# XGBClassifier정확도 : 0.8603
# LGBMClassifier정확도 : 0.8603
# 보팅결과 :  0.8659
