import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.python.keras.models import  load_model 
import matplotlib
from sklearn.preprocessing import LabelEncoder
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

path = './_data/test_amore_0718/' # ".은 현재 폴더"
# Amo1 = pd.read_csv(path + '아모레220718.csv' ,sep='\t',engine='python',encoding='CP949')
Amo = pd.read_csv(path + '아모레220718.csv',thousands=',',encoding='CP949')

Sam = pd.read_csv(path + '삼성전자220718.csv',thousands=',',encoding='CP949')

# print(Amo) #[3180 rows x 17 columns]
# print(Sam) #[3040 rows x 17 columns]
# print(Amo.head())
# print(Amo.describe().transpose())
# print(Amo['시가'].corr()) # 상관 계수 확인

# Amo["시가"].plot(figsize=(12,6)) 
# plt.show() # 'T (degC)'열의 전체 데이터 시각화

# plt.figure(figsize=(20,10),dpi=120)
# plt.plot(Amo['시가'][0:6*24*365],color="black",linewidth=0.2)
# plt.show() # 'T (degC)'열의 1년간 데이터 추이 시각화

# print(Amo.info()) #[3180 rows x 17 columns] objetct 14
print(Sam.info()) #(3040 rows x 17 columns] objetct 14


# Amo = Amo.drop([1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783], axis=0)

print(Sam.shape) #3037,17

# Sam.at[:1036, '시가'] =1
# print(Sam['시가'])
print(Amo) #2018/05/04
Amo.at[1035:,'종가'] = 0
print(Amo) #2018/05/04

'''
# Amo.index = pd.to_datetime(Amo['일자'],
#                             format = "%Y/%m/%d") 
# Sam.index = pd.to_datetime(Sam['일자'],
#                             format = "%Y/%m/%d") 
Amo['Date'] = pd.to_datetime(Amo['일자'])

Amo['year'] = Amo['Date'].dt.strftime('%Y')
Amo['month'] = Amo['Date'].dt.strftime('%m')
Amo['day'] = Amo['Date'].dt.strftime('%d')
print(Amo)
print(Amo.shape)
Sam['Date'] = pd.to_datetime(Sam['일자'])

Sam['year'] = Sam['Date'].dt.strftime('%Y')
Sam['month'] = Sam['Date'].dt.strftime('%m')
Sam['day'] = Sam['Date'].dt.strftime('%d')



Sam = Sam[Sam['시가'] < 100000] #[1035 rows x 17 columns]
print(Sam.shape)
print(Sam)
Amo = Amo[Amo['시가'] > 100] #[1035 rows x 17 columns]
print(Amo.shape)
print(Amo) #2018/05/04


# data에 index 열을 Date Time에 연,월,일,시간,분,초를 각각 문자열로 인식해 대체합니다.
# print(Amo.info()) #(420551, 15) DatetimeIndex: 3180 entries, 2022-07-18 to 2009-09-01
cols = ['year','month','day']
for col in cols:
    le = LabelEncoder()
    Amo[col]=le.fit_transform(Amo[col])
    Sam[col]=le.fit_transform(Sam[col])
print(Amo) 
print(Amo.info())

Amo = Amo.rename(columns={'Unnamed: 6':'증감량'})
Sam = Sam.rename(columns={'Unnamed: 6':'증감량'})


Amo = Amo.sort_values(by=['일자'],axis=0,ascending=True)
Sam = Sam.sort_values(by=['일자'],axis=0,ascending=True)
print(Amo)

Amo = Amo.drop(['일자'], axis=1)
Sam = Sam.drop(['일자'], axis=1)

print(Amo) #[70067 rows x 15 columns] 중복되는 값은 제거한다 행이 70091->에서 70067로 줄어든 것을 확인

# Amo1 = Amo.drop([ '전일비', '금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'],axis=1) #axis는 컬럼 
# Sam1 = Sam.drop([ '전일비', '금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'],axis=1) #axis는 컬럼 

Amo1 = Amo[[ '시가', '고가', '저가', '종가','year','month','day']]
Sam1 = Sam[[ '시가', '고가', '저가', '종가','year','month','day']]
Amo2 = Amo1['시가']


print(Amo1) #[1035 rows x 8 columns]
print(Sam1) #[1035 rows x 8 columns]




def generator(data, window, offset):
    gen = data.to_numpy() #데이터 프레임을 배열객체로 반환
    X = []
    
    for i in range(len(gen)-window-offset): # 420522
        row = [[a] for a in gen[i:i+window]] #행
        X.append(row)
        
        
    return np.array(X)
WINDOW = 5
OFFSET = 24
print(Amo1.shape) #(1031, 3, 10)

aaa = generator(Amo1,WINDOW,OFFSET) #x1를 위한 데이터 drop 제외 모든 amo에 데이터
bbb = generator(Amo2,WINDOW,OFFSET) #Y를 위한 시가만있는 데이터
x1 = aaa[:,:-1]
y = bbb[:,-1]
ccc = generator(Sam1,WINDOW,OFFSET) #x2를 위한 데이터 drop 제외 모든 Sam에 데이터
x2 = ccc[:,:-1]

print(x1,x1.shape) #(1006, 4, 1, 7)
print(x2,x2.shape) #(1006, 4, 1, 7)

print(y,y.shape) #(1006, 1)
y = y.reshape(1006,)
print(y.shape) #(1006, )



#시계열 데이터의 특성 상 연속성을 위해서 train_test_split에 셔플을 배제하기 위해
#위 명령어로 정의한다.suffle을 False로 놓고 해도 될지는 모르겠다.
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import LSTM,Dense,Dropout,Reshape,Conv1D
from tensorflow.python.keras.layers import Input
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test =train_test_split(x1,x2,y,shuffle=False,train_size=0.75)
x1_train = x1_train.reshape(754,4,7)
x1_test = x1_test.reshape(252,4,7)
x2_train = x2_train.reshape(754,4,7)
x2_test = x2_test.reshape(252,4,7)
print(x1_train,x1_train.shape) #(754, 4, 1,7)
print(x1_test,x1_test.shape) #(252, 4, 1,7)
print(x2_train,x2_train.shape) # (754, 4, 1,7)
print(x2_test,x2_test.shape) # (252, 4, 1,7)
print(y_train.shape) #(754,)


print(x2.shape) #(252, 2, 7) [:,-5:-2] #(252, 2, 7)

# 2-1. 모델구성1
input1 = Input(shape=(4,7)) #(N,2)
dense1 = LSTM(80,activation='relu',name='jk1')(input1)
drop1 = Dropout(0.35)(dense1)
dense2 = Dense(48,activation='relu',name='jk2')(drop1) 
drop2 = Dropout(0.35)(dense2)
dense3 = Dense(65,activation='relu',name='out_jk1')(drop2)
drop3 = Dropout(0.45)(dense3)
output1 = Dense(1,activation='relu',name='out_jk2')(drop3)

#2-2. 모델구성2
input2 = Input(shape=(4,7)) #(N,2)
dense4 = LSTM(99,activation='relu',name='jk101')(input2)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(84,activation='relu',name='jk103')(drop4) 
drop5 = Dropout(0.3)(dense5)
dense6 = Dense(64,activation='relu',name='jk104')(drop5) 
drop6 = Dropout(0.3)(dense6)
dense7 = Dense(474,activation='relu',name='jk105')(drop6) 
drop7 = Dropout(0.3)(dense7)
output2 = Dense(1,activation='relu',name='out_jk3')(drop7)


from tensorflow.python.keras.layers import concatenate,Concatenate
merge1 = concatenate([output1,output2],name= 'mg1')
merge2 = Dense(32,activation='relu',name='mg2')(merge1)
merge3= Dropout(0.3)(merge2)
merge4 = Dense(16,activation='relu',name='mg3')(merge3)
merge5= Dropout(0.4)(merge4)
merge6= Dense(16,activation='linear',name='mg4')(merge5)
last_output = Dense(1,name='last')(merge6)
# model = Model(inputs=[input1,input2], outputs=last_output)
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

#3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k24_', date, '_', filename])
                    )
model.compile(loss='mae', optimizer='Adam')

model.fit([x1_train,x2_train], y_train, 
          validation_split=0.20, 
          epochs=10,verbose=2
          ,batch_size=100
          ,callbacks=[earlyStopping])
model.save_weights("./_save/keras46_1_save_weights4.h5")
model = load_model("./_ModelCheckPoint/K24/k24_0719_0046_0269-138391.5469.hdf5")


#4. 평가,예측
loss = model.evaluate([x1_test,x2_test], y_test)
print("loss :",loss)
print("====================")
x1= x1_test[-3:-2]
x2= x2_test[-3:-2]

y_predict = model.predict([x1,x2])
print("0719자 시가 :",y_predict)


# id_pred_df.to_csv(path + 'sample_submission.csv', index=True)

# loss : 71854.9453125
# ====================
# 0719자 시가 : [[103615.4]]
'''