from grpc import AuthMetadataContext
from tensorflow.keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import math
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from sklearn.metrics import r2_score, accuracy_score
train_datagen = ImageDataGenerator(
    rescale=1./255,             #스케일링
    horizontal_flip=True,       #수평으로 뒤집어준다
    vertical_flip=True,         #수직으로 뒤집어준다 
    width_shift_range=0.1,      #가로로 움직이는 범위          
    height_shift_range=0.1,     #세로로 움직이는 범위
    rotation_range=5,           #이미지 회전           
    zoom_range=1.2,             #임의 확대/축소 범위
    shear_range=0.7,            #임의 전단 변환 (shearing transformation) 범위 #짜부~
    fill_mode='nearest'         #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식    
)

#평가데이터이기때문에 이미지 증폭은 하면 X
test_datagen = ImageDataGenerator(
    rescale=1./255
)
\
#D:\project\actor\actor\actor
xy_data = train_datagen.flow_from_directory(
    'C:/study/_data/test/',
    target_size=(70,70),
    batch_size=5000, 
    class_mode='categorical', 
    # color_mode='grayscale',
    shuffle=False           
) #Found 4369 images belonging to 30 classes.

#D:\project\test\test\test
# test = test_datagen.flow_from_directory(
#      'C:/study/_data/test/',
#      target_size=(75,75)
# )
x = xy_data[0][0]
y = xy_data[0][1]


print(x.shape)   #(4369, 75, 75, 3)
print(y.shape)   #(4369, 30)


x_train, x_test,y_train,y_test = train_test_split(x,y,
                                  train_size=0.8, shuffle=True, random_state=30)


np.save('d:/study_data/_save/_npy/keras106_8_train_x.npy', arr=x_train)
np.save('d:/study_data/_save/_npy/keras106_8_train_y.npy', arr=y_train)
np.save('d:/study_data/_save/_npy/keras106_8_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras106_8_test_y.npy', arr=y_test)

# x_train = np.load('d:/study_data/_save/_npy/keras103_5_train_x.npy')
# y_train = np.load('d:/study_data/_save/_npy/keras103_5_train_y.npy')
# y_train  = np.load('d:/study_data/_save/_npy/keras105_5_train_y.npy')
# xy_train = np.load('d:/study_data/_save/_npy/keras105_5_train_x.npy')