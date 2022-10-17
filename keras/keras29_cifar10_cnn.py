from random import random
from turtle import shape
from keras.datasets import mnist, fashion_mnist , cifar10
from sympy import python 
from tensorflow.python.keras.layers import Conv2D, MaxPool2D,Dense,Flatten ,Input,Dropout
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

plt.style.use('seaborn-white') 
#.1 데이터

(x_train, y_train, ),(x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)
print(y_test[5]) #[6]

np.random.seed(777)
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','sheep','truck']
sample_size=9
random_idx = np.random.randint(60000, size=sample_size)

plt.figure(figsize=(5, 5))
for i, idx in enumerate(random_idx):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(class_names[int(y_train[i])])
    plt.show()
    
    x_mean = np.mean(x_train, axis=(0, 1 ,2))
    x_std = np.std(x_train, axis=(0, 1, 2))
    
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,train_size=0.7)
    
    print(x_train.shape)
    print(y_train.shape)
    
    print(x_val.shape)
    print(y_val.shape)
    
    print(x_test.shape)
    print(y_test.shape)
    
# (35000, 32, 32, 3)
# (35000, 1)
# (15000, 32, 32, 3)
# (15000, 1)
# (10000, 32, 32, 3)
# (10000, 1)


 #2. 모델구성 
def model_build():
     model = Sequential()
     
     input= Input(shape=(32,))
     
     output = Conv2D(filter=32, kernel_size=3, padding='same',activation='relu')(input)
     output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)
     
     output = Conv2D(filter=64, kernel_size=3, padding='same',activation='relu')(output)
     output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    
     output = Conv2D(filter=128, kernel_size=3, padding='same',activation='relu')(output)
     output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)
    
     output = Flatten()(output)
     output = Dense(256, activation = 'relu')(output)
     output = Dense(128, activation = 'relu')(output)
     output = Dense(10, activation = 'softmax')(output)
     
     model = Model(input=[input], outputs=output)
     
     model.compile(optimizer = 'Adam'(learning_rate=1e-4),
                   loss ='sparse_catrgorical_crossentropy',
                   metrics=['accuracy',])
     return model
 
 
model = model_build()
model.summary()
 