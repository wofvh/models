from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
import numpy as np
import time
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM
import pandas as pd

#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target


df = pd.DataFrame (x, columns=[['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']])

print(datasets.feature_names)

print(x.shape,y.shape) #(569, 30) (569,)

x = x.reshape(569,30,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

print(x_train.shape, x_test.shape)  #(455, 30, 1) (114, 30, 1)


#2. 모델구성

model = Sequential()
model.add(LSTM(units=100 ,return_sequences=True,
               activation='relu', input_shape =(30,1)))
model.add(LSTM(units=100 ,return_sequences=False,
               activation='relu' ))
model.add(Dense(25, activation='relu'))    
model.add(Dense(15, activation='relu'))    
model.add(Dense(35, activation='relu'))    
model.add(Dense(45, activation='relu'))    
model.add(Dense(50, activation='relu'))    
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

date = datetime.datetime.now()

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)




end_time = time.time() - start_time

#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict.shape)
print(y_test.shape)

y_predict = to_categorical(y_predict)
y_predict = np.argmax(y_predict, axis= 1)

print(y_test.shape)
print(y_predict.shape)
print(y_test)
print(y_predict)

from sklearn.metrics import accuracy_score, r2_score

acc= accuracy_score(y_test, y_predict)
print('loss : ' , loss)
print('acc스코어 : ', acc) 
print("걸린시간 : ", end_time)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot =True, cbar =True)
plt.show()