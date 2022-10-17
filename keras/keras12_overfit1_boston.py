from tabnanny import verbose
from tracemalloc import start
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7, shuffle=True,
                                                    random_state=50 )

#print (x)
#print (y)
#print(x.shape, y.shape)   #(506. 13) (506,)

#print(datasets.feature_names)
#print(datasets.DESCR)

#[실습] 아래를 완성하기
# 1. train 0.7
# 2. R2 0.8 이상


#2.모델구성
model = Sequential() #순차적 
model.add(Dense(512, input_dim=13))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

import time
#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x,y, epochs=50, batch_size=1, verbose=1,validation_split = 0.2)

end_time = time.time()               

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
plt.title("천재")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc='upper right')
plt.show()