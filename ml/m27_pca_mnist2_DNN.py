#[실습]
#아까 4가지로 모델을 만들기
#784 개 DNN으로 만들기 (최상의 성능인거 // 0.966이상)과 비교

#time 체크/fit에서 하고 


#1.나의최고의dnn
# time=?
# acc=???

#2.나의최고의cnn
#time = ??
#acc = ??

#3.PCA 0.95
#time = ??
#acc = ??

#4.PCA 0.99
#time = ??
#acc = ??

#5.PCA 0.999
#time = ??
#acc = ??

#6.PCA 1.0
#time = ??
#acc = ??

#데이터1 

#############################################
import numpy as np 
from sklearn.decomposition import PCA 
from keras.datasets import mnist
from unittest import result
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing ,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__)
import warnings
import time
warnings.filterwarnings(action="ignore")


# start = time.time()
# (x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
# print(x_train.shape , x_test.shape)   #(60000, 28, 28) (10000, 28, 28)
# x = np.append(x_train,x_test, axis=0)
# print(x.shape)  #(70000, 28, 28)
# x = x.reshape(70000,784)
# print(x.shape)

#################################################
#[실습]
#pca를 통해 0.95 이상인 n_components는 몇개 ? 

start = time.time() # 시작 시간 체크
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) # (70000, 784)
print(x.shape) # (70000, 784)
y= np.append(y_train, y_test) # (70000,)

pca = PCA(n_components=486) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = pca.fit_transform(x) # x를 pca로 변환한다.
pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.
cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.

print('n_components=', 783, ':') # 중요도를 이용해 주요하지 않은 변수를 제거한다.
print(np.argmax(cumsum >= 0.95)+1) #154
print(np.argmax(cumsum >= 0.99)+1) #331
print(np.argmax(cumsum >= 0.999)+1) #486
print(np.argmax(cumsum+1)) #712

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66, stratify=y) 


from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 

model = RandomForestClassifier()

# tree = method ='gpu_hist', predictor = 'gpu_predictor',
# gpu_id = 0,




# model = XGBClassifier()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
end = time.time() # 종료 시간 체크

print('실행 시간 :', end-start)
print('accuracy :', result)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

end = time.time()- start


# 실행 시간 : 79.73798394203186
# accuracy : 0.9415


# 실행 시간 : 104.25226783752441
# accuracy : 0.9382142857142857


# 실행 시간 : 179.41868042945862
# accuracy : 0.9084285714285715