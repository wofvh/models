import select
from selectors import SelectSelector
from sklearn.datasets import load_breast_cancer , load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
import time 
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression 

#1.데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(442, 10) (442,)

x_train , x_test , y_train , y_test = train_test_split( x,y,
    shuffle=True, random_state=72 ,train_size=0.8
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kflod = KFold(n_splits=5 , shuffle=True, random_state=123)

#2모델
# model = XGBRegressor(
#                     )

model = LinearRegression()

model.fit(x_train,y_train,)
        #    eval_set=[(x_train,y_train),(x_test,y_test)],
        #    #eval_set=[(x_test,y_test)],
        #    eval_metric ='error')

results =model.score(x_test, y_test)
print('최종 점수:',results )

y_predict = model.predict(x_test)
acc= r2_score(y_test, y_predict)
print("진짜 최종TEST점수:", acc)


'''
print(model.feature_importances_)

thresholds = model.feature_importances_
print("===============================")

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape,select_x_test.shape)
    
    selection_model= XGBRegressor()
    selection_model.fit(select_x_train,y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("thresh=%.3F, N=%d, R2:%.2f%%"
          %(thresh, select_x_train.shape[1],score*100))
'''


# 최종 점수: -1.5732111511642377
# 진짜 최종TEST점수: -1.5732111511642377
# [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819
#  0.06012432 0.09595273 0.30483875 0.06629313]