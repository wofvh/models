from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRFRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings(action="ignore")


#1.데이터 
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569, 30) (569,)

x_train , x_test , y_train , y_test = train_test_split( x,y,
    shuffle=True, random_state=123 ,train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kflod = StratifiedKFold(n_splits=5 , shuffle=True, random_state=123)

#2모델
# model = XGBClassifier(random_state=123,
#                       n_setimators=1000,
#                       n_estimators=100,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gamma=1,
#                     )
#3. 훈련
# model.fit(x_train,y_train,
#           early_stopping_rounds = 50, eval_set=[(x_train,y_train),(x_test,y_test)],
#            #eval_set=[(x_test,y_test)],
#            eval_metric ='error')
##################불러오기/// 2모델 /3훈련

import joblib
path = "d:/study_data/_save/_xg/"
model = joblib.load(path + "m39_picklel_save.dat",)




#4. 평가 ,예측
results =model.score(x_test, y_test)
print('최종 점수:',results )

y_predict = model.predict(x_test)
acc= accuracy_score(y_test, y_predict)
print("진짜 최종TEST점수:", acc)

