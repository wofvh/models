import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Sequential 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

path = "C:\dacon\\tatanit/"

train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")


print(train.head())
print(test.head())
print(train.shape)  #(891, 12)
print(test.shape)   #(418, 11)

print(train.info())
print(train.describe()) #컬럼 평균치
print("=============================")

print(train['Survived'].value_counts()) #죽은사람 549 #산사람 342

# print(train['Survived'].value_counts().plot.bar())
# plt.show()

print(train.groupby('Pclass').mean()['Survived']) #groupby

# 1    0.629630
# 2    0.472826
# 3    0.242363

# plt.subplot(1, 2, 1)
# train.groupby('Pclass').mean()['Survived'].plot.bar()
# plt.ylabel('Survived Rate')

# plt.subplot(1,2,2)
# train['Pclass'].value_counts().plot.bar()
# plt.ylabel('Count')
# plt.show()

# plt.subplot(1,2,1)
# train.groupby('Sex').mean()['Survived'].plot.bar()
# plt.ylabel('Survived Rate')

# plt.subplot(1,2,2)
# train['Sex'].value_counts().plot.bar()
# plt.ylabel('count')

# plt.show()

# train['Age'].plot.hist(bins = 30)
# plt.show()

# sns.kdeplot(train[train['Survived']==0]['Age'], label = "Die")
# sns.kdeplot(train[train['Survived']==1]['Age'], label = "Survived")
# plt.legend() 

# plt.show()

print(train['SibSp'].value_counts())

# 0    608
# 1    209
# 2     28
# 4     18
# 3     16
# 8      7
# 5      5

# plt.subplot(1,2,1)
# train.groupby('SibSp').mean()['Survived'].plot.bar()
# plt.ylabel('Survived Rate')

# plt.subplot(1,2,2)
# train.groupby('Parch').mean()['Survived'].plot.bar()
# plt.ylabel('Survived Rate')
# plt.show()

# plt.figure(figsize=(20,12))

# plt.subplot(2,1,1)
# sns.kdeplot(train.loc[train['Survived']==0,'Fare'], label = 'Dead')
# sns.kdeplot(train.loc[train['Survived']==1,'Fare'], label = 'Survived')
# plt.legend()

# plt.show()  #요금이 비쌀수록 생존률이 높다

# plt.subplot(1, 2, 1)
# train.groupby('Embarked').mean()['Survived'].plot.bar()
# plt.ylabel('Survived Rate')

# plt.subplot(1 ,2, 2)
# train['Embarked'].value_counts().plot.bar()
# plt.ylabel('count')
# plt.show()

print(test.info())

#  #   Column       Non-Null Count  Dtype               
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object

#컬럼중 이름 티켓 승무원 데이터는 사용하기 어렵기에 변수를 없애준다 
#승객 아이디도 의미없어서 index 역활이기에 삭제

train.drop(['Name','Ticket','Cabin','PassengerId'], axis= 1 ,inplace= True)
test.drop(['Name','Ticket','Cabin','PassengerId'], axis= 1 ,inplace= True)

print(train.head())
# print(test.head())

#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  418 non-null    int64
#  1   Pclass       418 non-null    int64
#  2   Name         418 non-null    object
#  3   Sex          418 non-null    object
#  4   Age          332 non-null    float64
#  5   SibSp        418 non-null    int64
#  6   Parch        418 non-null    int64
#  7   Ticket       418 non-null    object
#  8   Fare         417 non-null    float64
#  9   Cabin        91 non-null     object
#  10  Embarked     418 non-null    object


#   #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Survived  891 non-null    int64
#  1   Pclass    891 non-null    int64
#  2   Sex       891 non-null    object
#  3   Age       714 non-null    float64
#  4   SibSp     891 non-null    int64
#  5   Parch     891 non-null    int64
#  6   Fare      891 non-null    float64
#  7   Embarked  889 non-null    object

train['Family_Size'] = train['SibSp'] + train['Parch']
test['Family_Size'] = test['SibSp'] + test['Parch']

print(train.head())

# train.groupby('Family_Size').mean()['Survived'].plot.bar()
# plt.show()

for row in range(train.shape[0]):
    if train.loc[row, 'Family_Size'] ==0:
        train.loc[row,'Family_Size'] = "S"
    elif (train.loc[row,'Family_Size'] >=1) & (train.loc[row,'Family_Size']<4):
        train.loc[row, 'Family_Size'] = 'M'
    else:
        train.loc[row,'Family_Size'] = "L"
# train.groupby('Family_Size').mean()['Survived'].plot.bar()
# plt.show()

for row in range(test.shape[0]):
    if test.loc[row, 'Family_Size'] ==0:
        test.loc[row,'Family_Size'] = "S"
    elif (test.loc[row,'Family_Size'] >=1) & (test.loc[row,'Family_Size']<4):
        test.loc[row, 'Family_Size'] = 'M'
    else:
        test.loc[row,'Family_Size'] = "L"

print(train.head())

print(train['Fare'].describe()) #Fare티켓 값 

train['Fare'] = np.log1p(train['Fare'])
test['Fare'] = np.log1p(test['Fare'])

plt.figure(figsize=(20,12))

# plt.subplot(2,1,1)
# sns.kdeplot(train.loc[train['Survived']==0, 'Fare'], label = 'Dead')
# sns.kdeplot(train.loc[train['Survived']==1, 'Fare'], label = 'Survived')
# plt.legend()

# plt.show()

print(train.head())
print(test.head())

#2. 모델구성
model = Sequential()
model.add(Dense(19, input_dim=8, activation='linear')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
model.add(Dense(18, activation='sigmoid'))               # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
model.add(Dense(19, activation='relu'))               # relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
model.add(Dense(25, activation='linear'))               
model.add(Dense(1, activation='sigmoid'))   
                                                                        
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])   # 이진분류에 한해 로스함수는 무조건 99퍼센트로 'binary_crossentropy'
                                      # 컴파일에있는 metrics는 평가지표라고도 읽힘


earlyStopping = EarlyStopping(monitor='val_loss', patience=400, mode='auto', verbose=1, 
                              restore_best_weights=True)        

                  #restore_best_weights false 로 하면 중단한 지점의 웨이트값을 가져옴 true로하면 끊기기 전의 최적의 웨이트값을 가져옴



model.fit(train, train, epochs=3000, batch_size=100,
                 validation_split=0.3,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(test, test)
y_predict = model.predict(test)


#### 과제 1 accuracy_score 완성 y 테스트는 반올림이 됫는데 y 프리딕트는 반올림이 안됫음 ######
print(y_predict)
y_predict = y_predict.round(0)
print(y_predict)


y_summit = model.predict(test)

print(y_summit)
print(y_summit.shape) # (418, 1)
y_summit = y_summit.round()
df = pd.DataFrame(y_summit)
print(df)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
y_summit = oh.fit_transform(df)
print(y_summit)
y_summit = np.argmax(y_summit, axis= 1)
submission_set = pd.read_csv(path + 'gender_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

print(submission_set)

submission_set['Survived'] = y_summit
print(submission_set)


submission_set.to_csv(path + 'submission.csv', index = True)


acc= accuracy_score(test, y_predict)
print('loss : ' , loss)
print('acc스코어 : ', acc) 
