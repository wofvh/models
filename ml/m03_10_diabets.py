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
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=77)

print(y_train.shape)     #(353,)
print(x_train.shape)     #(353, 10)



#2. 모델구성
# allAgorithms = all_estimators(type_filter='classifier')  #분류모델
allAgorithms = all_estimators(type_filter='regressor') #회기모델

# model = LinearRegression() 결과 acc: 0.5034051724671988
# model = LinearRegression()   결과 r2: 0.5034051724671988
# model = KNeighborsRegressor()  결과 r2: 0.4583895962258292
# model = DecisionTreeRegressor()  결과 r2: 0.15448404811171423
# model = RandomForestRegressor()  #결과 r2: 0.4819457161866536

###try except 에러가 떳을때 애러처리### 
for (name, algorithm) in allAgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
    
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test,y_predict)
        print(name, '의 정답률:', r2)
    except:
        # continue
        print(name,"은 안나온놈 !!!")

#fit

model.fit(x_train, y_train)

#4.평가,예측
results = model.score(x_test,y_test)

# print('결과 acc:', results)
print('결과 r2:', results)


