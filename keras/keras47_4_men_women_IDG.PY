# 실습
# 본인 사진으로 predict 하시오
# d:/study_data/_data/image/
import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

#1. 데이터

train = ImageDataGenerator(
    rescale=1./255,               # MinMax 스케일링과 같은 개념 
  )                               # 회전 축소 등으로 이미지에 여백이생겼을때 채우는 방법입니다.

# xydata = train.flow_from_directory(
#     'D:\study_data\_data\data',
#     target_size=(150,150),
#     class_mode='binary',
#     batch_size=500,
#     shuffle=True,) # 경로 및 폴더 설정

# print(xydata[0][0],xydata[0][0].shape) # (500, 150, 150, 3)

# x = xydata[0][0]
# y = xydata[0][1]
my = train.flow_from_directory(
    'D:\study_data\_data\image\men_women\me',
    target_size=(150,150),
    class_mode='binary',
    batch_size=1,
    shuffle=True,) # 경로 및 폴더 설정


# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=100,train_size=0.7)

# np.save('D:/study_data/_save/_npy/keras47_4_train_x.npy',arr=x_train)
# np.save('D:/study_data/_save/_npy/keras47_4_train_y.npy',arr=y_train)
# np.save('D:/study_data/_save/_npy/keras47_4_test_x.npy',arr=x_test)
# np.save('D:/study_data/_save/_npy/keras47_4_test_y.npy',arr=y_test)

np.save('D:/study_data/_save/_npy/keras_test_a.npy',arr=my[0][0])