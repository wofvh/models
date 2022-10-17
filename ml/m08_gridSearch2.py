import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
#GridSearchCV   격자 탐색 cross validation
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    test_size=0.25,
                                                    shuffle=True ,random_state=100)
# print(y_train,y_test)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {"C":[1, 10, 100, 1000],"kernel":["linear"],"degree":[3,4,5]},             
    {"C":[1, 10, 100],"kernel":["rbf"],"gamma":[0.001,0.0001]},               
    {"C":[1, 10, 100, 1000],"kernel":["sigmoid"],
     "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}]     

# 각 횟수를 병렬로 진행해 총 42번을  1회에 한다.
#rbf= Gaussian basis function RBF 뉴럴네트워크의 경우 각 데이터에 맞는 
# Kernel function을 이용하기에 비선형적이고, MLP보다 학습이 빠르다.

#2. 모델 구성
from sklearn.svm import LinearSVC,SVC


# model = LinearSVC() 
# model = LogisticRegression() 
# model = KNeighborsClassifier() 
# model = DecisionTreeClassifier() 
# model = SVC(C=1,kernel='linear',degree=3)
model = GridSearchCV(SVC(),parameters,cv=kfold,verbose=1,
                     refit=True,n_jobs=-1,) 

# Fitting 5 folds(kfold의 인수) for each of 42 candidates, totalling 210 fits(42*5)
# n_jobs=-1 사용할 CPU 갯수를 지정하는 옵션 '-1'은 최대 갯수를 쓰겠다는 뜻
#3. 컴파일,훈련
model.fit(x_train,y_train) 


# #4.  평가,예측

print("model_score :",model.score(x_test,y_test))
# model_score : 1.0

y_predict = model.predict(x_test)