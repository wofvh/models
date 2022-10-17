from unittest import result
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from tensorboard import summary
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x)
print(y)
print(x.shape, y.shape) #(150, 4) (150,)

print("y의 라벨값:" , np.unique(y)) #y의 라벨값: [0 1 2]


y = to_categorical(y)
print(y)
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, shuffle= True,
                                                    random_state=66 )
print(y_train)
print(y_test)



#2. 모델구성
model = Sequential() #순차적 
model.add(Dense(30, activation='relu', input_dim=4)) #sigmoid 0~1 로 분류함 0.5 기준으로 (반올림)
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))  #relu 히든레이어에서만 가능 
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))

np.argmax(x)
print(x)


#컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) #이중분류에서 categorical_crossentropy 

#이진분류 한해 로수함수는 무조건 99프로 binary_crossentropy
#binary_crossentropy (반올림)
from tensorflow.python.keras.callbacks import EarlyStopping 
earlystopping = EarlyStopping(monitor='val_loss',patience=10,mode='min', verbose=1,
              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlystopping],
                 verbose=1)

import tensorflow as tfimport 

#평가,예측
results = model.evaluate(x_test, y_test)
print('loss:',results[0])
print('accuracy', results[1])

# print("==============y_test[:5]===================")
# print(y_test[:5])
# y_pred = model.predict(x_test[:5])
# print("=================================")
# print(y_pred)
# print("=================================")

# print(y_predict)
# print(y_predict)
# y_predict = to_categorical(y_predict)
# print(y_test)
# print(y_predict)
print('=============================')

from sklearn.metrics import r2_score, accuracy_score
y_predict =model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)


acc = accuracy_score(y_test, y_predict)
print('acc스코어:', acc)

plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], marker='.', label = 'loss',color='red' )
plt.plot(hist.history['val_loss'], marker='.', label ='val_loss',color='blue' )
plt.grid()
plt.title("천재")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc='upper right')
plt.show()

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()