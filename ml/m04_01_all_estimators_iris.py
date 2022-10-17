from cv2 import Algorithm
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, RobustScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.linear_model import LogisticRegression,LinearRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier,KNeighborsTransformer    #
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor     # 
from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor  # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 
from sklearn.datasets import load_iris
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=77)

print(y_train.shape)     #(404, 13)v
print(x_train.shape)     #(404,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성

allAgorithms = all_estimators(type_filter='classifier')  #분류모델
# allAgorithms = all_estimators(type_filter='regressor') #회기모델

print('allAgorithms:', allAgorithms )
print('모델의 갯수:',len(allAgorithms)) #41
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),

###try except 에러가 떳을때 애러처리### 
for (name, algorithm) in allAgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
    
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(name, '의 정갑률:', acc)
    except:
        # continue
        print(name,"은 안나온놈 !!!")
        
# 모델의 갯수: 41
# AdaBoostClassifier 의 정갑률: 0.9
# BaggingClassifier 의 정갑률: 0.8666666666666667
# BernoulliNB 의 정갑률: 0.3
# CalibratedClassifierCV 의 정갑률: 0.7666666666666667
# CategoricalNB 의 정갑률: 0.26666666666666666
# ClassifierChain 은 안나온놈 !!!
# ComplementNB 의 정갑률: 0.7333333333333333
# DecisionTreeClassifier 의 정갑률: 0.8666666666666667
# DummyClassifier 의 정갑률: 0.26666666666666666
# ExtraTreeClassifier 의 정갑률: 0.9666666666666667
# ExtraTreesClassifier 의 정갑률: 0.8666666666666667
# GaussianNB 의 정갑률: 0.8666666666666667
# GaussianProcessClassifier 의 정갑률: 0.8333333333333334
# GradientBoostingClassifier 의 정갑률: 0.9
# HistGradientBoostingClassifier 의 정갑률: 0.8333333333333334
# KNeighborsClassifier 의 정갑률: 0.8333333333333334
# LabelPropagation 의 정갑률: 0.8666666666666667
# LabelSpreading 의 정갑률: 0.8666666666666667
# LinearDiscriminantAnalysis 의 정갑률: 0.9333333333333333
# LinearSVC 의 정갑률: 0.8666666666666667
# LogisticRegression 의 정갑률: 0.8
# LogisticRegressionCV 의 정갑률: 0.8666666666666667
# MLPClassifier 의 정갑률: 0.8
# MultiOutputClassifier 은 안나온놈 !!!
# MultinomialNB 의 정갑률: 0.5333333333333333
# NearestCentroid 의 정갑률: 0.8
# NuSVC 의 정갑률: 0.8666666666666667
# OneVsOneClassifier 은 안나온놈 !!!
# OneVsRestClassifier 은 안나온놈 !!!
# OutputCodeClassifier 은 안나온놈 !!!
# PassiveAggressiveClassifier 의 정갑률: 0.8333333333333334
# Perceptron 의 정갑률: 0.8
# QuadraticDiscriminantAnalysis 의 정갑률: 0.9333333333333333
# RadiusNeighborsClassifier 의 정갑률: 0.5
# RandomForestClassifier 의 정갑률: 0.8666666666666667
# RidgeClassifier 의 정갑률: 0.7666666666666667
# RidgeClassifierCV 의 정갑률: 0.7333333333333333
# SGDClassifier 의 정갑률: 0.8333333333333334
# SVC 의 정갑률: 0.8333333333333334
# StackingClassifier 은 안나온놈 !!!
# VotingClassifier 은 안나온놈 !!!

# (tf282gpu) C:\study>C:/anaconda3/envs/tf282gpu/python.exe c:/study/ml/m04_all_estimators.py
# (120,)
# (120, 4)
# allAgorithms: [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), ('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>), ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), ('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>), ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), ('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>), ('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>), ('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), ('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), ('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>), ('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), ('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>), ('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', <class 'sklearn.linear_model._logistic.LogisticRegression'>), ('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), ('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>), ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), ('NearestCentroid', <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>), ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>), ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), ('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), ('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>), ('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), ('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>), ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), ('SGDClassifier', <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), ('SVC', <class 'sklearn.svm._classes.SVC'>), ('StackingClassifier', <class 'sklearn.ensemble._stacking.StackingClassifier'>), ('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)]
# 모델의 갯수: 41
# AdaBoostClassifier 의 정갑률: 0.8666666666666667
# BaggingClassifier 의 정갑률: 0.8666666666666667
# BernoulliNB 의 정갑률: 0.3
# CalibratedClassifierCV 의 정갑률: 0.7666666666666667
# CategoricalNB 의 정갑률: 0.26666666666666666
# ClassifierChain 은 안나온놈 !!!
# ComplementNB 의 정갑률: 0.7333333333333333
# DecisionTreeClassifier 의 정갑률: 0.9
# DummyClassifier 의 정갑률: 0.26666666666666666
# ExtraTreeClassifier 의 정갑률: 0.9
# ExtraTreesClassifier 의 정갑률: 0.8666666666666667
# GaussianNB 의 정갑률: 0.8666666666666667
# GaussianProcessClassifier 의 정갑률: 0.8333333333333334
# GradientBoostingClassifier 의 정갑률: 0.9
# HistGradientBoostingClassifier 의 정갑률: 0.8333333333333334
# KNeighborsClassifier 의 정갑률: 0.8333333333333334
# LabelPropagation 의 정갑률: 0.8666666666666667
# LabelSpreading 의 정갑률: 0.8666666666666667
# LinearDiscriminantAnalysis 의 정갑률: 0.9333333333333333
# LinearSVC 의 정갑률: 0.8666666666666667
# LogisticRegression 의 정갑률: 0.8
# LogisticRegressionCV 의 정갑률: 0.8666666666666667
# MLPClassifier 의 정갑률: 0.8333333333333334
# MultiOutputClassifier 은 안나온놈 !!!
# MultinomialNB 의 정갑률: 0.5333333333333333
# NearestCentroid 의 정갑률: 0.8
# NuSVC 의 정갑률: 0.8666666666666667
# OneVsOneClassifier 은 안나온놈 !!!
# OneVsRestClassifier 은 안나온놈 !!!
# OutputCodeClassifier 은 안나온놈 !!!
# PassiveAggressiveClassifier 의 정갑률: 0.8666666666666667
# Perceptron 의 정갑률: 0.8
# QuadraticDiscriminantAnalysis 의 정갑률: 0.9333333333333333
# RadiusNeighborsClassifier 의 정갑률: 0.5
# RandomForestClassifier 의 정갑률: 0.8666666666666667
# RidgeClassifier 의 정갑률: 0.7666666666666667
# RidgeClassifierCV 의 정갑률: 0.7333333333333333
# SGDClassifier 의 정갑률: 0.8
# SVC 의 정갑률: 0.8333333333333334
# StackingClassifier 은 안나온놈 !!!
# VotingClassifier 은 안나온놈 !!!

