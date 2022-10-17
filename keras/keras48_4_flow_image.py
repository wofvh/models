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

augument_size = 10  #<<<설명  사진에서 augument 숫자만큼 빼겟다 
randidx = np.random.randint(x_train.shape[0], size=augument_size) #if [0] 일때(6000,28,28)
print(x_train.shape[0])    #60000
print(randidx)             #[56989 41218  7342 43513 29552 20513 37206 20878 39946 24317]  7342 56989     
print(np.min(randidx),np.max(randidx)) #1 59998
print(type(randidx))  #<class 'numpy.ndarray'> base 리스트 형태 태 

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()


x_train = x_train.reshape(60000,28,28,1) 

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)  

print(x_test.shape)        #(10000, 28, 28, 1)
print(x_train.shape)        #(60000,28 ,28 1)

x_augumented = x_augumented.reshape(x_augumented.shape[1],
                                    x_augumented.shape[2],1)

x_augumented = train_datagen.flow(x_augumented,y_augumented,
                                  batch_size = augument_size,
                                  shuffle=False).next()[0]

# print(x_augumented.shape)  #(10, 28, 28, 1)
# print(x_test.shape)        #(10000, 28, 28, 1)
# print(y_test.shape)        #(10000,)
# print(x_train.shape)        #(60000, 28, 28, 1)
# print(y_train.shape)        #((60000,)

x_train =np.concatenate((x_train, x_augumented))
y_train =np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) #(60010, 28, 28, 1) (60010,)
# print(x_train.shape[0])  #60000

#실습
#x_augumented 10개와 x_train 10 개를 비교하는 이미지 출력 할것!

print(x_train[0].shape)    #(28, 28)
print(x_train[0].reshape(28*28).shape)  #(784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape)
#(100, 28, 28,1)
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape)

x_train = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28,1 ),
    (np.zeros(20)),
    batch_size= 20,
    shuffle=True).next()

print(x_train)

######################. next() 사용#######################
# # print(x_train)
# # <keras.preprocessing.image.NumpyArrayIterator object at 0x00000230E56D2EB0>
# print(x_train[0])           #x와 y가 모두 포함
# print(x_train[0].shape)  #(10, 28, 28, 1)
# print(x_train[1].shape)  #(10,)

# #######################. next() 미사용#######################
# print(x_train)
# # <keras.preprocessing.image.NumpyArrayIterator object at 0x00000230E56D2EB0>
# print(x_train[0])           #x와 y가 모두 포함
# print(x_train[0][0].shape)  #(100, 28, 28, 1)
# print(x_train[0][1].shape)  #(100,)


import matplotlib.pyplot as plt
plt.figure(figsize=(2,7))
for i in range(20):
    plt.subplot(2,10, i+1)
    plt.axis('off')
    plt.imshow(x_train[0][i], cmap='gray')    #next 사용
    # plt.imshow(x_train[0][0][i], cmap='gray') #next미사용
plt.show()