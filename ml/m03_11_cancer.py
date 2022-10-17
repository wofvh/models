import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.svm import LinearSVC,LinearSVR
from sklearn import datasets                 #분류 & 회기
from sklearn.linear_model import LogisticRegression,LinearRegression   #LogisticRegression  로지스틱 분류모델 
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor    #
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor     # 
from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor  # decisiontreeclassfier 가 랜덤하게 앙상블로 역김 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score

#1.데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=77)

print(y_train.shape)     #(353,)
print(x_train.shape)     #(353, 10)



#2. 모델구성\
# model = LogisticRegression()      #  결과 r2: 0.9473684210526315
# model = KNeighborsClassifier()      # 결과 r2: 0.9385964912280702
# model = DecisionTreeClassifier()     # 결과 r2: 0.9385964912280702
model = RandomForestClassifier()     # 결과 r2: 0.9385964912280702



#3.fit
model.fit(x_train, y_train)

#4.평가,예측
results = model.score(x_test,y_test)

# print('결과 acc:', results)
print('결과 r2:', results)

