import numpy as np
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) #(178, 13) (178,)

# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets['target']
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 



print(x.shape, y.shape) #178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15,shuffle=True ,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
# scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(y_train,y_test)
#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron ,LogisticRegression 
#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #공부하자 
from sklearn.ensemble import RandomForestClassifier #공부하자 
from sklearn.linear_model import LinearRegression 

def models(model):
    if model == 'knn':
        mod = KNeighborsClassifier()
    elif model == 'svc':
        mod = SVC()
    elif model == 'tree':
        mod =  DecisionTreeClassifier()
    elif model == 'forest':
        mod =  RandomForestClassifier()
    elif model == 'linear':
        mod =  LinearRegression ()    
    elif model == 'linearSVC':
        mod =  LinearSVC ()       
    return mod
model_list = ['knn', 'svc',  'tree', 'forest','linear','linearSVC']
cnt = 0
empty_list = [] #empty list for progress bar in tqdm library
for model in (model_list):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)
    #Training
    clf.fit(x_train, y_train) 
    #Predict
    result = clf.score(x_test,y_test)
    pred = clf.predict(x_test) 
    print('{}-{}'.format(model,result))
    
# knn-0.8888888888888888
# svc-0.9629629629629629
# tree-0.8148148148148148
# forest-1.0
# linear-0.8878003396163971
# linearSVC-0.9629629629629629