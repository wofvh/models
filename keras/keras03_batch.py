# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# #1. 데이터
# import numpy as np
# x = np. array([1,2,3,5,4]) #array 변수에 여러 개의 값을 저장할때
# y = np. array([1,2,3,4,5]) 

# #실습 6

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model = Sequential()   
# model.add(Dense(4, input_dim=1)) 
# model.add(Dense(230)) 
# model.add(Dense(123))
# model.add(Dense(143))
# model.add(Dense(190))
# model.add(Dense(1))

# model.compile(loss='mae',optimizer='adam') #(mse 평균제곱오차) (loss 오차) compile (Optimizer(최적화)
# model.fit(x, y, epochs=64, batch_size=1)

# loss = model.evaluate(x, y) #평가값을 로스에 넣는다 #(evaliate 평가)
# print('loss : ',loss)

# result = model.predict([6])
# print('6의 예측값 : ',result )

# # loss :  0.44082245230674744
# # 6의 예측값 :  [[9]]


import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
import numpy as np
x = np.array([range(1,1039)])

print(x)