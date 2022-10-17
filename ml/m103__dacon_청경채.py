from http.cookiejar import LWPCookieJar
import pandas as pd
import numpy as np
import glob
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.metrics import accuracy_score,r2_score
import os

path = 'D:/study_data/dacon/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

train_data, label_data = aaa(train_input_list, train_target_list) #, False)

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)

#2.모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(64,2 ,input_shape=(1440,37)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))


#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(train_data, label_data, epochs=100)


#4.평가,예측
results= model.evaluate(train_data, val_input_list)

loss = model.evaluate(label_data,train_data)
result = model.predict((val_target_list))  #평가예측에서 똑같이 맟춰서
print('loss:',loss)
print('predict결과:',result)  #RNN input_shape 에서 들어간 차원을 야함

print('accuracy : ', results)

#5. 파일저장

import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_vegi/test_target")
with zipfile.ZipFile("submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
