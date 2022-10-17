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
import time

#1.데이터
datasets = load_boston()

x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7, shuffle=True,
                                                    random_state=50 )



#2.모델구성
model = Sequential() #순차적 
model.add(Dense(6, input_dim=13))
model.add(Dense(89))
model.add(Dense(225))
model.add(Dense(225))
model.add(Dense(285))
model.add(Dense(155))
model.add(Dense(228))
model.add(Dense(92))
model.add(Dense(1))

#3.컴파일 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping 
earlystopping = EarlyStopping(monitor='loss',patience=100, mode='min', verbose=1,
              restore_best_weights=True)

              
start_time = time.time()

hist = model.fit(x_train,y_train, epochs=1000, batch_size=1, verbose=1,
                 callbacks=[earlystopping], validation_split = 0.2)


end_time = time.time() - start_time            

#평가,예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2스코어 :', r2 )



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
