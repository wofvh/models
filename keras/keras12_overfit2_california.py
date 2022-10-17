from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1.데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

start_time = time.time()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=66)

end_time = time.time()   

#2.모델
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(21))
model.add(Dense(22))
model.add(Dense(23))
model.add(Dense(24))
model.add(Dense(1))

import time
#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x,y, epochs=11, batch_size=1, verbose=1,validation_split = 0.2)

#평가,예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)



print("---------------------------------")
print(hist) #<tensorflow.python.keras.callbacks.History object at 0x0000020DCD274340>
print("---------------------------------")
print(hist.history)
print("---------------------------------")
print(hist.history['loss'])
print("---------------------------------")
print(hist.history['val_loss'])

print("걸린시간:", end_time )

plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], marker='.', label = 'loss',color='red' )
plt.plot(hist.history['val_loss'], marker='.', label ='val_loss',color='blue' )
plt.grid()
plt.title("켈리포니아")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc='upper right')
plt.show()

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

