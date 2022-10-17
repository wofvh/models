from unittest import result
import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import true
from tensorboard import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets['data']
y = datasets['target']

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#                                                     train_size=0.8, shuffle= True,
#                                                     random_state=66 )

n_splits = 5
# Kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)
Kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)


# #2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier    #
from sklearn.tree import DecisionTreeClassifier       # 
from sklearn.ensemble import RandomForestClassifier   # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 
allAgorithms = all_estimators(type_filter='classifier')  #분류모델
# allAgorithms = all_estimators(type_filter='regressor') #회기모델
print('allAgorithms:', allAgorithms )
print('모델의 갯수:',len(allAgorithms)) #41
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),

###try except 에러가 떳을때 애러처리### 
for (name, algorithm) in allAgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=Kfold)
        print('ACC :', scores,'\n cross_val_score:', round(np.mean(scores),4)) # cross_val_score: 0.9667
        
    except:
        # continue
        print(name,"은 안나온놈 !!!")
    
#컴파일 훈련

# 모델의 갯수: 41
# ACC : [0.63333333 0.93333333 1.         0.9        0.96666667] 
#  cross_val_score: 0.8867
# ACC : [0.93333333 0.96666667 1.         0.86666667 0.96666667] 
#  cross_val_score: 0.9467
# ACC : [0.3        0.33333333 0.3        0.23333333 0.3       ]
#  cross_val_score: 0.2933
# ACC : [0.9        0.83333333 1.         0.86666667 0.96666667] 
#  cross_val_score: 0.9133
# ACC : [0.9        0.93333333 0.93333333 0.9        1.        ]
#  cross_val_score: 0.9333
# ClassifierChain 은 안나온놈 !!!
# ACC : [0.66666667 0.66666667 0.7        0.6        0.7       ]
#  cross_val_score: 0.6667
# ACC : [0.96666667 0.96666667 1.         0.9        0.93333333] 
#  cross_val_score: 0.9533
# ACC : [0.3        0.33333333 0.3        0.23333333 0.3       ]
#  cross_val_score: 0.2933
# ACC : [0.93333333 0.96666667 1.         0.83333333 0.93333333]
#  cross_val_score: 0.9333
# ACC : [0.93333333 0.96666667 1.         0.86666667 0.96666667] 
#  cross_val_score: 0.9467
# ACC : [0.96666667 0.9        1.         0.9        0.96666667]
#  cross_val_score: 0.9467
# ACC : [0.96666667 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score: 0.96
# ACC : [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
#  cross_val_score: 0.9667
# ACC : [0.86666667 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score: 0.94
# ACC : [0.96666667 0.96666667 1.         0.9        0.96666667]
#  cross_val_score: 0.96
# ACC : [0.93333333 1.         1.         0.9        0.96666667]
#  cross_val_score: 0.96
# ACC : [0.93333333 1.         1.         0.9        0.96666667]
#  cross_val_score: 0.96
# ACC : [1.  1.  1.  0.9 1. ]
#  cross_val_score: 0.98
# ACC : [0.96666667 0.96666667 1.         0.9        1.        ]
#  cross_val_score: 0.9667
# ACC : [1.         0.96666667 1.         0.9        0.96666667]
#  cross_val_score: 0.9667
# ACC : [1.         0.96666667 1.         0.9        1.        ] 
#  cross_val_score: 0.9733
# ACC : [0.96666667 0.96666667 1.         0.93333333 1.        ] 
#  cross_val_score: 0.9733
# MultiOutputClassifier 은 안나온놈 !!!
# ACC : [0.96666667 0.93333333 1.         0.93333333 1.        ] 
#  cross_val_score: 0.9667
# ACC : [0.93333333 0.9        0.96666667 0.9        0.96666667] 
#  cross_val_score: 0.9333
# ACC : [0.96666667 0.96666667 1.         0.93333333 1.        ] 
#  cross_val_score: 0.9733
# OneVsOneClassifier 은 안나온놈 !!!
# OneVsRestClassifier 은 안나온놈 !!!
# OutputCodeClassifier 은 안나온놈 !!!
# ACC : [0.76666667 0.83333333 0.86666667 0.7        0.93333333] 
#  cross_val_score: 0.82
# ACC : [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 
#  cross_val_score: 0.78
# ACC : [1.         0.96666667 1.         0.93333333 1.        ] 
#  cross_val_score: 0.98
# ACC : [0.96666667 0.9        0.96666667 0.93333333 1.        ] 
#  cross_val_score: 0.9533
# ACC : [0.93333333 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score: 0.9533
# ACC : [0.86666667 0.8        0.93333333 0.7        0.9       ] 
#  cross_val_score: 0.84
# ACC : [0.86666667 0.8        0.93333333 0.7        0.9       ] 
#  cross_val_score: 0.84
# ACC : [0.8        0.63333333 0.66666667 0.76666667 0.96666667] 
#  cross_val_score: 0.7667
# ACC : [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
#  cross_val_score: 0.9667
# StackingClassifier 은 안나온놈 !!!
# VotingClassifier 은 안나온놈 !!!

# (tf282gpu) C:\study>