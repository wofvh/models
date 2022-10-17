import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.metrics import r2_score
from xgboost import XGBClassifier,XGBRegressor
from sklearn.feature_selection import SelectFromModel
#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(10886, 11)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv',#예측에서 쓸거야!!
                       index_col=0)
            
print(test_set)
print(test_set.shape) #(6493, 8) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 


###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
print(train_set.shape) #(10886,11)


x = train_set.drop([ 'casual', 'registered','count'],axis=1) #axis는 컬럼 


print(x.columns)
print(x.shape) #(10886, 8)

y = train_set['count']

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor #공부하자 

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=1234)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

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
