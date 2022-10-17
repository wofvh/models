import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE # imblearn 다운로드 해야함 pip install imblearn 
import sklearn as sk 
print("사이킷런: ", sk.__version__)  #사이킷런:  1.1.2

#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)  #(178, 13) (178,)
print(type(x))           #<class 'numpy.ndarray'>
print(np.unique(y, return_counts = True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

print(pd.Series(y).value_counts()) #< pandas 로 바꾸자
print(y)

# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0/ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1/ 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# #  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-25]
y=  y[:-25]

print(pd.Series(y).value_counts())

# 1    71
# 0    59
# 2    8

x_train, x_test, y_train, y_test = train_test_split(
    x , y, train_size=0.75, shuffle=True, random_state=123, stratify= y
) 
print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    6

#2, 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 컴파일 .훈련
model.fit(x_train,y_train)


#4. 평가 ,예측
y_predict  = model.predict(x_test)

score = model.score(x_test,y_test)  
# print("model.score:", score)
print('acc_score:',accuracy_score (y_test,y_predict))
print('f1_score(macro): ',f1_score(y_test,y_predict , average="macro")) #F1_score 은 2진분류에서 많이 사용 다중분류에서 average="macro" 사용
# print('f1_score(micro): ',f1_score(y_test,y_predict , average="micro")) #F1_score 은 2진분류에서 많이 사용 


# 기본결과
# acc_score: 0.972972972972973
# f1_score(macro):  0.9797235023041475


# acc_score: 0.9777777777777777
# f1_score(macro):  0.9797235023041475

#y, x 40 개씩줄였을때
# acc_score: 0.9428571428571428
# f1_score(macro):  0.8596176821983273

print("================================SMOTE 적용후=============================")
smote = SMOTE(random_state=123)                             #SMOTE 증폭 
x_train, y_train = smote.fit_resample(x_train, y_train)


print(pd.Series(y_train).value_counts())

# 0    53
# 1    53
# 2    53

model = RandomForestClassifier()
model.fit(x_train,y_train)

#4. 평가예측
y_predict  = model.predict(x_test)

score = model.score(x_test,y_test)  
# print("model.score:", score)
print('acc_score:',accuracy_score (y_test,y_predict))
print('f1_score(macro): ',f1_score(y_test,y_predict , average="macro")) #F1_score 은 2진분류에서 많이 사용 다중분류에서 average="macro" 사용
# print('f1_score(micro): ',f1_score(y_test,y_predict , average="micro")) #F1_score 은 2진분류에서 많이 사용 
