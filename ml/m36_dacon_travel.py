from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
import numpy as np 
#1. 데이터
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)


test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)

# print(train_set.shape,test_set.shape) (1955, 19) (2933, 18)                       
# Data columns (total 19 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Age                       1861 non-null   float64  MonthlyIncome
#  1   TypeofContact             1945 non-null   object # 빈도로 메꾸기 
#  2   CityTier                  1955 non-null   int64 
#  3   DurationOfPitch           1853 non-null   float64 앞뒤행으로 
#  4   Occupation                1955 non-null   object
#  5   Gender                    1955 non-null   object
#  6   NumberOfPersonVisiting    1955 non-null   int64
#  7   NumberOfFollowups         1942 non-null   float64  
#  8   ProductPitched            1955 non-null   object
#  9   PreferredPropertyStar     1945 non-null   float64
#  10  MaritalStatus             1955 non-null   object
#  11  NumberOfTrips             1898 non-null   float64
#  12  Passport                  1955 non-null   int64
#  13  PitchSatisfactionScore    1955 non-null   int64
#  14  OwnCar                    1955 non-null   int64
#  15  NumberOfChildrenVisiting  1928 non-null   float64
#  16  Designation               1955 non-null   object
#  17  MonthlyIncome             1855 non-null   float64 직급?나이?
#  18  ProdTaken                 1955 non-null   int64
######### 결측치 채우기 (클래스별 괴리가 큰 컬럼으로 평균 채우기)
print(train_set.shape)
index1 = train_set[train_set['MonthlyIncome'] >= 90000].index
index2 = train_set[train_set['MonthlyIncome'] <= 2000].index
index3 = test_set[test_set['MonthlyIncome'] <= 5000].index
print(index1,index2,index3)




# train_set = train_set.drop([190,605,1339],inplace=True)
# test_set = test_set.drop(index3,index=True)


train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
# print(train_set.isnull().sum()) #(1955, 19)
# print(train_set[train_set['MonthlyIncome'].notnull()].groupby(['Designation'])['MonthlyIncome'].mean())
train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
# print(train_set.describe) #(1955, 19)
# print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())
train_set['DurationOfPitch'].fillna(train_set.groupby('Occupation')['DurationOfPitch'].transform('mean'), inplace=True)
test_set['DurationOfPitch'].fillna(test_set.groupby('Occupation')['DurationOfPitch'].transform('mean'), inplace=True)
# print(train_set[train_set['NumberOfFollowups'].notnull()].groupby(['NumberOfChildrenVisiting'])['NumberOfFollowups'].mean())
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
# print(train_set[train_set['PreferredPropertyStar'].notnull()].groupby(['Occupation'])['PreferredPropertyStar'].mean())
train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
# train_set['AgeBand'] = pd.cut(train_set['Age'], 5)
# 임의로 5개 그룹을 지정
# print(train_set['AgeBand'])
# [(17.957, 26.6] < (26.6, 35.2] < (35.2, 43.8] <
# (43.8, 52.4] < (52.4, 61.0]]
combine = [train_set,test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 26.6, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 26.6) & (dataset['Age'] <= 35.2), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 35.2) & (dataset['Age'] <= 43.8), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 43.8) & (dataset['Age'] <= 52.4), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 52.4, 'Age'] = 4
# train_set = train_set.drop(['AgeBand'], axis=1)
# print(train_set[train_set['NumberOfTrips'].notnull()].groupby(['DurationOfPitch'])['PreferredPropertyStar'].mean())
train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
# print(train_set[train_set['NumberOfChildrenVisiting'].notnull()].groupby(['MaritalStatus'])['NumberOfChildrenVisiting'].mean())
train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
# print(train_set.isnull().sum()) 
# print("================")
# print(test_set.isnull().sum()) 
cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
# print(train_set['TypeofContact'])
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))



Age_out_index= outliers(train_set['Age'])[0]
print("이상치의 위치 :",Age_out_index)

TypeofContact_out_index= outliers(train_set['TypeofContact'])[0]
CityTier_out_index= outliers(train_set['CityTier'])[0]
DurationOfPitch_out_index= outliers(train_set['DurationOfPitch'])[0]
Gender_out_index= outliers(train_set['Gender'])[0]
NumberOfPersonVisiting_out_index= outliers(train_set['NumberOfPersonVisiting'])[0]
NumberOfFollowups_out_index= outliers(train_set['NumberOfFollowups'])[0]
ProductPitched_index= outliers(train_set['ProductPitched'])[0]
PreferredPropertyStar_out_index= outliers(train_set['PreferredPropertyStar'])[0]
MaritalStatus_out_index= outliers(train_set['MaritalStatus'])[0]
NumberOfTrips_out_index= outliers(train_set['NumberOfTrips'])[0]
Passport_out_index= outliers(train_set['Passport'])[0]
PitchSatisfactionScore_out_index= outliers(train_set['PitchSatisfactionScore'])[0]
OwnCar_out_index= outliers(train_set['OwnCar'])[0]
NumberOfChildrenVisiting_out_index= outliers(train_set['NumberOfChildrenVisiting'])[0]
Designation_out_index= outliers(train_set['Designation'])[0]
MonthlyIncome_out_index= outliers(train_set['MonthlyIncome'])[0]



lead_outlier_index = np.concatenate((Age_out_index,
                                     TypeofContact_out_index,
                                     CityTier_out_index,
                                     DurationOfPitch_out_index,
                                     Gender_out_index,
                                     NumberOfPersonVisiting_out_index,
                                     NumberOfFollowups_out_index,
                                     ProductPitched_index,
                                     PreferredPropertyStar_out_index,
                                     MaritalStatus_out_index,
                                     NumberOfTrips_out_index,
                                     Passport_out_index,
                                     PitchSatisfactionScore_out_index,
                                     OwnCar_out_index,
                                     NumberOfChildrenVisiting_out_index,
                                     Designation_out_index,
                                     MonthlyIncome_out_index
                                     ),axis=None)
print(len(lead_outlier_index)) #577
# print(lead_outlier_index)

lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
# print(train_set_clean)

x = train_set_clean.drop(['ProdTaken'], axis=1)

y = train_set_clean['ProdTaken']
# print(x.shape,y.shape) #(1528, 18) (1528,)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,shuffle=True,random_state=1234)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis() 
# lda.fit(x_train,y_train)
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)
# test_set = lda.transform(test_set)
n_splits = 5 

kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)

parameters = {'gamma': [0.1], 'learning_rate': [0.1,0.2,0.3,0.4], 
             'max_depth': [6,7,8], 'min_child_weight': [1], 'n_estimators': [100,200], 'subsample': [1]}

# parameters = [
#     {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
#      'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
#     {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
#      'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
#     ]    

xgb = XGBClassifier(random_state=103)

model = GridSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)
import time
start_time =time.time()
model.fit(x_train,y_train)
end_time = time.time()-start_time
# model.score(x_test,y_test)
results = model.score(x_test,y_test)
print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('model.socre : ',results)
print('걸린 시간 : ',end_time)
y_summit = model.predict(test_set)
y_summit = np.round(y_summit,0)
submission = pd.read_csv(path + 'v,#예측에서 쓸거야!!
                      )
#
submission['ProdTaken'] = y_summit
# submission = submission.fillna(submission.mean())
# submission = submission.astype(int)
submission.to_csv('test22.csv',index=False)
