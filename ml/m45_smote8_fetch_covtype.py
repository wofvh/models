import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel   # 모델을 선택.
from imblearn.over_sampling import SMOTE  # SMOTE install 필요
import sklearn as sk
print(sk.__version__)

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)      # (178, 13) (178,)
print(type(x))              # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True))         # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.Series(y).value_counts())
print(y)

# x = x[:-23]
# y = y[:-23]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123,
                                                    shuffle=True,
                                                    train_size=0.75,
                                                    stratify=y
                                                    )

print(pd.Series(y_train).value_counts())

# 모델저장
import pickle
path = "d:/study_data/_save/_xg/"
model = pickle.load(open(path + "m39_fetch_covtype_save.dat", "rb"))

#4. 평가
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
# print('결과:',score)
print('acc:', accuracy_score(y_test,y_predict))
print('f1_macro:',f1_score(y_test,y_predict, average='macro'))  

# 증폭 전 