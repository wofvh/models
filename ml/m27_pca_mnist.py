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
warnings.filterwarnings(action="ignore")

(x_train,_),(x_test,_) = mnist.load_data()


print(x_train.shape , x_test.shape)   #(60000, 28, 28) (10000, 28, 28)

x = np.append(x_train,x_test, axis=0)
print(x.shape)  #(70000, 28, 28)

x = x.reshape(70000,784)
print(x.shape)

#################################################
#[실습]
#pca를 통해 0.95 이상인 n_components는 몇개 ? 

pca = PCA(n_components=784)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)
print(cumsum)


print(np.argmax(cumsum >= 0.95)+1)
print(np.argmax(cumsum >= 0.99)+1)
print(np.argmax(cumsum >= 0.999)+1)
print(np.argmax(cumsum >= 1.0)+1)


import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
 

# 0.95
# 0.99
# 0.999
# np.argmax


# x = datasets.data
# y = datasets.target
# print(x.shape,y.shape )    #(506, 13) (506,)

# pca = PCA(n_components=10)
# x = pca.fit_transform(x)
# print(x.shape)              #(506, 2)


# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))


# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

