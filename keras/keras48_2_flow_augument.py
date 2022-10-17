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
augument_size = 40000  #<<<설명  사진에서 augument 숫자만큼 빼겟다 
randidx = np.random.randint(x_train.shape[0], size=augument_size) #if [0] 일때(6000,28,28)
print(x_train.shape[0])    #60000
print(randidx)             #[13729 20404 58580 ... 10711  8123 30104]
print(np.min(randidx),np.max(randidx)) #1 59998
print(type(randidx))  #<class 'numpy.ndarray'> base 리스트 형태 

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()
print(x_augumented.shape)  #(40000, 28, 28)
print(y_augumented.shape)  #(40000,) 

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2],1)

x_augumented = train_datagen.flow(x_augumented,y_augumented,
                                  batch_size = augument_size,
                                  shuffle=False).next()[0]

print(x_augumented)
print(x_augumented.shape)

x_train =np.concatenate((x_train, x_augumented))
y_train =np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape)
# print(x_train.shape[0])  #60000


print(x_train[0].shape)    #(28, 28)
print(x_train[0].reshape(28*28).shape)  #(784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape)
#(100, 28, 28,1)
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28,1 ),
    (np.zeros(augument_size)),
    batch_size= augument_size,
    shuffle=True).next()

######################. next() 사용#######################
print(x_data)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x00000230E56D2EB0>
print(x_data[0])           #x와 y가 모두 포함
print(x_data[0].shape)  #(100, 28, 28, 1)
print(x_data[1].shape)  #(100,)

# #######################. next() 미사용#######################
# print(x_data)
# # <keras.preprocessing.image.NumpyArrayIterator object at 0x00000230E56D2EB0>
# print(x_data[0])           #x와 y가 모두 포함
# print(x_data[0][0].shape)  #(100, 28, 28, 1)
# print(x_data[0][1].shape)  #(100,)


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')    #next 사용
    # plt.imshow(x_data[0][0][i], cmap='gray') #next미사용
plt.show()
