from unittest import result
import numpy as np
import py
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes 
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split,KFold,cross_val_score ,StratifiedKFold
from sklearn.metrics import accuracy_score

#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test , y_train,y_test = train_test_split(x,y,
                                                    train_size = 0.8, shuffle=True,
                                                    random_state=72)
print(y_test.shape)

n_splits = 5 #다섯 번씩 모든 데이터를 훈련해준다고 지정 

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#Kfold 모든 데이터를 버리는거 없이 나누어줌 


# allAgorithms = all_estimators(type_filter='classifier')  #분류모델
allAgorithms = all_estimators(type_filter='regressor') #회기모델
print('allAgorithms:', allAgorithms )
print('모델의 갯수:',len(allAgorithms)) #41

#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

for (name, algorithm) in allAgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print('ACC:', scores, ' \n cross_val_score:',round(np.mean(scores),5))
        
    except:
        print(name,"안 나온놈!!!")
#컴파일 훈련

#   warnings.warn(
# ACC: [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996]
#  cross_val_score: -1.20857
# ACC: [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873]
#  cross_val_score: 0.48416
# ACC: [0.46143523 0.49174877 0.48686617 0.34528933 0.52538972]  
#  cross_val_score: 0.46215
# ACC: [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626]  
#  cross_val_score: 0.33414
# ACC: [-0.06757503 -1.13368033  0.07691653 -0.01874053  0.51241126]  
#  cross_val_score: -0.12613
# ACC: [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
#  -9.58335111e-03]
#  cross_val_score: -0.00331
# ACC: [0.3534533  0.47281314 0.48656369 0.36842632 0.43607511]  
#  cross_val_score: 0.42347
# RegressorChain 안 나온놈!!!
# ACC: [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091]
#  cross_val_score: 0.42118
# ACC: [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194]
#  cross_val_score: 0.48845
# C:\anaconda3\envs\tf282gpu\lib\site-packages\sklearn\linear_model\_stochastic_gradient.py:1225: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
#   warnings.warn("Maximum number of iteration reached before "
# C:\anaconda3\envs\tf282gpu\lib\site-packages\sklearn\linear_model\_stochastic_gradient.py:1225: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
#   warnings.warn("Maximum number of iteration reached before "
# C:\anaconda3\envs\tf282gpu\lib\site-packages\sklearn\linear_model\_stochastic_gradient.py:1225: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
#   warnings.warn("Maximum number of iteration reached before "
# C:\anaconda3\envs\tf282gpu\lib\site-packages\sklearn\linear_model\_stochastic_gradient.py:1225: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
#   warnings.warn("Maximum number of iteration reached before "
# C:\anaconda3\envs\tf282gpu\lib\site-packages\sklearn\linear_model\_stochastic_gradient.py:1225: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
#   warnings.warn("Maximum number of iteration reached before "
# ACC: [0.39331533 0.44165303 0.46456853 0.32955095 0.41509034]  
#  cross_val_score: 0.40884
# ACC: [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ]  
#  cross_val_score: 0.15914
# StackingRegressor 안 나온놈!!!
# ACC: [0.49976368 0.46180298 0.55149153 0.34153357 0.51897392]  
#  cross_val_score: 0.47471
# ACC: [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679]
#  cross_val_score: 0.48765
# ACC: [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042]
#  cross_val_score: 0.00316
# VotingRegressor 안 나온놈!!!