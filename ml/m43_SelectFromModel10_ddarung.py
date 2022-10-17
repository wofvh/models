import select
from selectors import SelectSelector
from sklearn.datasets import load_breast_cancer , load_diabetes , load_iris ,fetch_california_housing,load_breast_cancer, load_wine,fetch_covtype
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
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer



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



x_train , x_test , y_train , y_test = train_test_split( x,y,
    shuffle=True, random_state=123 ,train_size=0.8
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kflod = KFold(n_splits=5 , shuffle=True, random_state=123)

#2모델
model = XGBRegressor(random_state=123,
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,
                    )

model.fit(x_train,y_train,
          early_stopping_rounds = 200, eval_set=[(x_train,y_train),(x_test,y_test)],
           #eval_set=[(x_test,y_test)],
           eval_metric ='error')

results =model.score(x_test, y_test)
print('최종 점수:',results )

y_predict = model.predict(x_test)
acc= r2_score(y_test, y_predict)
print("진짜 최종TEST점수:", acc)

print(model.feature_importances_)

thresholds = model.feature_importances_
print("===============================")
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape,select_x_test.shape)
    
    selection_model= XGBRegressor(random_state=123,
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,)
    selection_model.fit(select_x_train,y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("thresh=%.3F, N=%d, R2:%.2f%%"
          %(thresh, select_x_train.shape[1],score*100))
