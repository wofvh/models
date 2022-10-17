#넘파이에서 부러러와서 모델구성 
#성능 비교 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=0.1,
    # # shear_range=0.7,
    # fill_mode='nearest'
)


x_train = np.load('D:/study_data/_save/_npy/keras49_1_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_1_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_1_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_1_test_y.npy')

print(x_train.shape)  # (3000, 28, 28, 1)
print(y_train.shape)  # (3000,)
print(x_test.shape)   #  (1000, 28, 28, 1)
print(y_test.shape) # (1000,)


#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Conv2D , Flatten,MaxPool2D

model = Sequential()
model.add(Conv2D(35,(2,2),input_shape = (28,28,1), activation='relu'))
model.add(Conv2D(64,(3,3),activation= 'relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(18,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()



#3.컴파일 훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#model.fit(cy_train[0][0], xy_train[0][1])# 배치를 최대로 자으면 이것도 가능 
hist = model.fit(x_train,y_train ,epochs=50 , validation_split=0.25)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ' ,loss[-1])
print('val_loss : ' ,val_loss[-1])
print('accuracy : ' ,accuracy[-1])
print('val_accuracy : ' ,val_accuracy[-1])

# loss :  -3.208187133264344e+19
# val_loss :  -3.49652812028581e+19
# accuracy :  0.1102222204208374
# val_accuracy :  0.09733333438634872



'''
############################################
import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets


#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,verbose=2,validation_split=0.25,batch_size=500)
# hist = model.fit_generator(x_train,y_train,epochs=2,
#                     validation_split=0.25,
#                     steps_per_epoch=32,
#                     validation_steps=4) # 배치가 최대 아닐 경우 사용

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
print('y_predict :', y_predict)

#증폭 후 
# loss : [0.35870376229286194, 0.8970000147819519] 
'''