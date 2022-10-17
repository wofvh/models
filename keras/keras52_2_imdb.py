from keras.datasets import imdb
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.layers import BatchNormalization
import pandas as pd

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)


print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train.shape, y_test.shape) # (25000,) (25000,)
print(np.unique(y_train, return_counts=True)) # [0, 1]
print(len(np.unique(y_train))) # 2

print(type(x_train), type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'>
print(len(x_train[0])) #218
print(len(x_train[1])) #189

print(max(len(i) for i in x_train)) 

print("뉴스기사의 최대길이 : " ,max(len(i) for i in x_train))
print("뉴스기사의 평균길이 : " , sum(map(len, x_train)) / len(x_train))
# 뉴스기사의 최대길이 :  2494
# 뉴스기사의 평균길이 :  238.71364

#전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                        #(8982,) -> (8982,100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')


# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) #(25000, 100) (25000,)               
print(x_test.shape, y_test.shape) #(25000, 100) (25000,)

#2. 모델 구성
             
model = Sequential()
                    #단어사전의 갯수  
# model.add(Embedding(input_dim=31, output_dim=11, input_length=5)) #embedding 에선 아웃풋딤이 뒤로 들어감
# model.add(Embedding(input_dim=31, output_dim=10)) # 인풋렝쓰는 모를 경우 안넣어줘도 자동으로 잡아줌
# model.add(Embedding(31, 10))
# model.add(Embedding(31, 10, 5)) # error
model.add(Embedding(10000, 20, input_length=100))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()             


#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1, 
                              restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=500 ,validation_split=0.2,callbacks=[es])



#4. 평가, 예측

acc = model.evaluate(x_test, y_test)[1]
print('acc : ', acc)
