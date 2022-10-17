import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from tensorboard import summary
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.model_selection import KFold, cross_val_score , GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV

# 1.데이터
datasets = load_iris()


x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.2, shuffle= True,
                                                    random_state=1234 )
n_splits = 5 #다섯 번씩 모든 데이터를 훈련해준다고 지정 

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

Parameters = [ 
            {"C":[1, 10, 100,1000], "kernel":["linear"],"degree":[3, 4, 5]},        #12
            {"C":[1, 10 ,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},     #6
            {"C":[1, 10,100,1000], "kernel":["sigmoid"],                        #24
            "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}                    #총42
]

# #2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier    #
from sklearn.tree import DecisionTreeClassifier       # 
from sklearn.ensemble import RandomForestClassifier   # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 
from sklearn.ensemble import RandomForestRegressor
# model = SVC(C=1, kernel="linear", degree=3)

model = HalvingGridSearchCV(RandomForestClassifier(),Parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)  #n_jobs cpu 갯수사용 정의 예) 1 cup 1 -1 cup8 


#컴파일 훈련
import time
start = time.time()

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ",model.best_params_)
print('best_score_:', model.best_score_)
print("model.score:", model.score(x_test,y_test))
end_time = time.time()
#평가,예측\ 
y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test, y_predict))
# print('accuracy', results[1])
y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test, y_pred_best))
print("걸린시간:", round(end_time - start, 4))