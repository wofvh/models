
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline


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
# [418 rows x 8 columns] (891, 9) (418, 8)

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

model = make_pipeline(StandardScaler(),LinearRegression())

model.fit(x_train,y_train)
print("기본 스코어 : ", model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold, scoring= 'r2')
print("cv:",scores)
print("기냥cv 엔빵:",np.mean(scores))

#2.모델구성

############################PolynomialFeatures후 ########################################

# pf = PolynomialFeatures(degree=2, include_bias = False) #차수는 2로 설정
pf = PolynomialFeatures(degree=2,) #차수는 2로 설정
xp = pf.fit_transform(x) 
print(xp.shape)  #(506, 105)

x_train,x_test,y_train,y_test = train_test_split(xp,
                                                 y,random_state=1234, train_size=0.8,)

# 2.모델구성
model = make_pipeline(StandardScaler(),LinearRegression())

model.fit(x_train,y_train)

print("폴리 스코어 : ", model.score(x_test, y_test))

scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 나눈 값 : ", np.mean(scores))


# 기본 스코어 :  0.3557820067614683
# cv: [0.41443668 0.27470532 0.45103367 0.36644746 0.39899381]
# 기냥cv 엔빵: 0.3811233878228498
# (891, 45)
# 폴리 스코어 :  0.47337895782554174
# 폴리 CV :  [0.32536364 0.28668924 0.35305015 0.46276106 0.00166175]
# 폴리 CV 나눈 값 :  0.2859051682011763