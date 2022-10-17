# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 

# #.1 데이터

# bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
#                 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
#                 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

# bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
#                 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
#                 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# plt.scatter(bream_length,bream_weight)
# plt.scatter(smelt_length,smelt_weight)
# plt.xlabel('length')
# plt.ylabel("weight")
# plt.show()

# weigth = bream_weight + smelt_weight
# length = bream_length + smelt_length

# fish_data = [[l,w] for l,w in zip(length,weigth)]

# print(fish_data)  

# fish_target = [1] * 35 + [0] * 14
# print(fish_target)

# #.2 모델구성
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# kn = KNeighborsClassifier(n_neighbors=17) 

# #.3 훈련
# kn.fit(fish_data,fish_target)

# kn.predict([[30,600]])

# #.4 평가, 예측
# # kn.score(fish_data, fish_target)
# results = kn.score(fish_data, fish_target)
# print('결과:', results)

# for n in range(5,50):
#     kn.n_neighbors = n 
#     score = kn.score(fish_data,fish_target)
#     if score < 1:
#         print(n, score)
#         break

######################## 두번째 문제 !!################################################
from operator import index
from re import L
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import concatenate, Concatenate
#.1 데이터

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0 ,9.8, 10.5,
                10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0,6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# print (np.column_stack(([1,2,3],[4,5,6])))


fish_data = np.column_stack((fish_length,fish_weight))


print(np.ones(49))

fish_target = np.concatenate((np.ones(35),np.zeros(14)))

print(fish_target)

train_input, test_input,train_target, test_target = train_test_split(fish_data,fish_target,random_state=42, stratify= fish_target)

# print(train_input.shape, test_input.shape)
# print(train_target.shape, test_target.shape)

print(test_target)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

model = KNeighborsClassifier()

model.fit(train_input,train_target)
model.score(test_input, test_target)
print(model.score(test_input, test_target))

print(model.predict([[25,150]]))

distances, indexes = model.kneighbors([[25,150]])

print(train_input[indexes])

# [[[ 25.4 242. ]
#   [ 15.   19.9]
#   [ 14.3  19.7]
#   [ 13.   12.2]
#   [ 12.2  12.2]]]

print(distances)

mean =np.mean(train_input,axis=0)
std = np.std(train_input,axis=0)

print(mean, std)

# [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]
# [ 27.29722222 454.09722222] [  9.98244253 323.29893931]

train_scaled = (train_input - mean) / std


import matplotlib.pyplot as plt
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(25, 150, marker='^')  #<<삼각형으로 나타낼때 씀'^' https://matplotlib.org/stable/api/markers_api.html <자세한 내용
# plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D') 
# plt.xlim((0,1000))  #<<x축의 범위를 지정해주는 함수 (x 랑 y 의 범위축이 너무 다르다) 인치로 재는 거와 센치로 재는 느낌 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()








































