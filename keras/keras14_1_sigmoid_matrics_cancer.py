import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score,accuracy_score
import matplotlib.pyplot as plt

#1.데이터
datasets = load_breast_cancer()

# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9, shuffle=True,
                                                    random_state=50 )

#print(x.shape, y.shape) #(569, 30) (569,)


#2.모델구성
model = Sequential() #순차적 
model.add(Dense(6, activation='linear', input_dim=30)) #sigmoid 0~1 로 분류함 0.5 기준으로 (반올림)
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))   #sigmoid 사용해보기 
model.add(Dense(100, activation='sigmoid'))  #relu 히든레이어에서만 가능 
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy','mse'])
#이진분류 한해 로수함수는 무조건 99프로 binary_crossentropy
#binary_crossentropy (반올림)

from tensorflow.python.keras.callbacks import EarlyStopping 
earlystopping = EarlyStopping(monitor='loss',patience=500,mode='min', verbose=1,
              restore_best_weights=True)

              
start_time = time.time()

hist = model.fit(x_train,y_train, epochs=1000, batch_size=100, verbose=1,
                 callbacks=[earlystopping], validation_split = 0.2)

end_time = time.time() - start_time            

#평가,예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)


y_predict = model.predict(x_test)
y_predict = y_predict.round(0)

#######[과제 accuracy_score 완성]###########
acc= accuracy_score(y_test, y_predict)
print('acc_score:', acc)
#r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2 )


# print("---------------------------------")
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x0000020DCD274340>
# print("---------------------------------")
# print(hist.history)
# print("---------------------------------")
# print(hist.history['loss'])
# print("---------------------------------")
# print(hist.history['val_loss'])

print("걸린시간:", end_time )

plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], marker='.', label = 'loss',color='red' )
plt.plot(hist.history['val_loss'], marker='.', label ='val_loss',color='blue' )
plt.grid()
plt.title("천재")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc='upper right')
plt.show()

# loss: [0.2297472357749939, 0.8771929740905762, 0.07352643460035324]
# acc_score: 0.8771929824561403