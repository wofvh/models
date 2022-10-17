 #2 결과를 뛰어넘어랏 
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


start = time.time() # 시작 시간 체크
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) # (70000, 784)
print(x.shape)#(70000, 784)
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


parameters = [
    {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
     'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
    {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
     'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
   
    ]  

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 

model = GridSearchCV(RandomForestClassifier(),parameters,verbose=1,
                     refit=True,n_jobs=-1) 

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
