from unittest import result
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
augument_size = 20  #<<<설명  사진에서 augument 숫자만큼 빼겟다 
randidx = np.random.randint(x_train.shape[0], size=augument_size) #if [0] 일때(6000,28,28)
print(x_train.shape[0])    #60000

print(randidx.shape[0])    #[0]40000      #[13729 20404 58580 ... 10711  8123 30104]

print(np.min(randidx),np.max(randidx)) #2 59999
print(type(randidx))  #<class 'numpy.ndarray'> base 리스트 형태 

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()
print(x_augumented.shape)  #(40000, 28, 28)
print(y_augumented.shape)  #(40000,) 

x_train = x_train.reshape(60000,28,28,1) 
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

# print(x_test.shape)   #(10000, 28, 28, 1)
# print(x_train.shape)  #(60000, 28, 28, 1)
# print(y_test.shape)   #(10000,)
# print(y_train.shape)  #(60000,)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2],1)

import time
start_time = time.time()

x_augumented = train_datagen.flow(x_augumented,y_augumented,
                                  batch_size = augument_size,
                                  save_to_dir='D:/study_data/_temp',
                                  shuffle=False).next()[0]
end_time = time.time() - start_time
print(augument_size,'개 증폭에 걸린시간: ',round(end_time,3 ),'초' )


# print(x_augumented)
# print(x_augumented.shape)    #(40000, 28, 28, 1)
# print(y_augumented.shape)    #(40000,)

# x_train =np.concatenate((x_train, x_augumented))
# y_train =np.concatenate((y_train, y_augumented))  #가로두개의 의미 

# print(x_train.shape) #(100000, 28, 28, 1)
# print(y_train.shape) #(100000,)

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Conv2D , Flatten,MaxPool2D
model = Sequential()
model.add(Conv2D(35,(2,2),input_shape = (28,28,1), activation='relu'))
model.add(Conv2D(64,(3,3),activation= 'relu'))
model.add(Flatten())
model.add(Dense(26,activation='relu'))
model.add(Dense(28,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(Dense(29,activation='relu'))
model.add(Dense(1,activation='relu'))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      '))
model.summary()


#3.컴파일 훈련
model.compile(loss= 'mae',optimizer='adam', metrics=['accuracy'])

#model.fit(cy_train[0][0], xy_train[0][1])# 배치를 최대로 자으면 이것도 가능 
hist = model.fit_generator(x_train,y_train ,epochs=30,batch_size=100,
                 validation_split=0.2,
                 verbose=2)


#4. 평가,예측
results = model.evaluate(x_test, y_test)
print("loss :",results)
y_predict = model.predict(x_test)
print('predict',y_predict[-1])

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7,7, i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][i], cmap='gray')    #next 사용
#     # plt.imshow(x_data[0][0][i], cmap='gray') #next미사용
# plt.show()