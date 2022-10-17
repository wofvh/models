from bitarray import test
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets

#############이미지 수치화 or 증폭가능############## 
train_datagen = ImageDataGenerator(
    rescale=1./225,  # rescale다른 처리 전에 데이터를 곱할 값입니다.1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다
    horizontal_flip=True, # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다
    vertical_flip=True,  #수직방향으로 뒤집기를 한다
    width_shift_range=0.1, # width_shift그림 을 height_shift수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    height_shift_range=0.1, #지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동시킨다. 예를 들어 0.1이고 전체 높이가 100이면, 10픽셀 내외로 상하 이동시킨다. 
    rotation_range=5,   # rotation_range사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
    zoom_range=1.2,   # zoom_range내부 사진을 무작위로 확대하기 위한 것입니다.
    shear_range=0.7,  # shear_range무작위로 전단 변환 을 적용하기 위한 것입니다.
    fill_mode='nearest'
)

# rotation_range사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
# width_shift그림 을 height_shift수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
# rescale다른 처리 전에 데이터를 곱할 값입니다. 원본 이미지는 0-255의 RGB 계수로 구성되지만 이러한 값은 모델이 처리하기에는 너무 높기 때문에(주어진 일반적인 학습률) 1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다. 요인.
# shear_range무작위로 전단 변환 을 적용하기 위한 것입니다.
# zoom_range내부 사진을 무작위로 확대하기 위한 것입니다.
# horizontal_flip이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다(예: 실제 사진).
# fill_mode회전 또는 너비/높이 이동 후에 나타날 수 있는 새로 생성된 픽셀을 채우는 데 사용되는 전략입니다.

########테스트 데이터는 증폭 안함#####
test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/image/brain/train/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',   #color_mode 안쓸경우 디폴드값은 컬러(3)
    shuffle = True,
    #Found 160 images belonging to 2 classes 160 데이터가 0~1로 데이터가 됬다
    #타겟싸이즈 맞춰야함 
)

xy_test = test_datagen.flow_from_directory(
    'D:/_data/image/brain/test/',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    shuffle = True,
    #Found 120 images belonging to 2 classes 0~1로 데이터가 됬다
)

# print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x000001F0E08D7A90>

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

print(xy_train[0])            #마지막 배치 
print(xy_train[0][0])  
print(xy_train[0][1])     #(5, 150, 150, 1) 

print(xy_train[0][0].shape, xy_train[0][1].shape) 

print(type(xy_train))     #반복자 DirectoryIterator
print(type(xy_train[0]))  #<class 'tuple'> 수정할수없는 Lsit
print(type(xy_train[0][0]))  #<class 'numpy.ndarray'>
print(type(xy_train[0][1]))  #<class 'numpy.ndarray'>


