from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRFRegressor
import time 
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier,XGBRFRegressor


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


def lgb_hamus(max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
     parameters ={ 
"n_estimators": [100,200,300,400,500,1000],  #디폴트 100/ 1~inf
"learning_rate" : [0.1, 0.2, 0,3, 0.5, 0.01, 0.001],# 디폴트 0.3/ 0~1
'max_depth' : [None,2,3,4,5,6,7,8,9,10] ,#디폴트 6 / 0~int/ 정수
'gamma' :[0, 1, 2, 3, 4, 5, 7, 10, 100], #디폴트 0 
'min_child_weight':[0, 0.01, 0.1, 0.5, 1, 5, 10,100],# 디폴드 1/0~int
'subsample': [0, 0.1, 0.2, 0.3,0.5, 0.7, 1], #디폴트 값 1
'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1], #디폴트 0.
'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1], #디폴트 1
'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1], #디폴트 1
'reg_alpha':[0, 0.1, 0.01, 0.001, 1, 2, 10], #디폴트0/0~int/L1 절대값 가중치 규제 /alpha 라고만해도됨
'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10], #디폴트1/0~int/L2 절대값 가중치 규제 /lambda 라고만해도됨
}



parameters = {"n_estimators" : [100,],
              "learning_rate" : [0.1],
              'max_depth' : [3],
              'gamma' :[1,],
              'min_child_weight':[1],
              'subsample': [1],
              'colsample_bytree':[0.5],
              'colsample_bylevel':[1],
              'colsample_bynode':[1],
              'reg_alpha':[0,],
              'reg_lambda':[1],
              } 

#2모델
xgb = XGBClassifier(random_state=123,
                    )
model = GridSearchCV(xgb, parameters, cv = kflod , n_jobs=8)

model.fit(x_train,y_train)

best_params = model.best_params_

print('최상의 매개변수:', model.best_params_)
print('최상의 점수:', model.best_score_)


#n_estimators
# 최상의 매개변수: {'n_estimators': 100}
# 최상의 점수: 0.9692307692307691
#learning_rate'
# 최상의 매개변수: {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수: 0.9648351648351647
#max_depth':
# 최상의 매개변수: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수: 0.9692307692307691
#'gamma':
# 최상의 매개변수: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수: 0.9692307692307691
#'min_child_weight': 
#최상의 매개변수: {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100}
#최상의 점수: 0.9736263736263737
#subsample': 
# 최상의 매개변수: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.5}
# 최상의 점수: 0.9736263736263737
# 'colsample_bytree': 0.5
# 최상의 매개변수: {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수: 0.9736263736263737

# reg_alpha
# 최상의 매개변수: {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'subsample': 1}
# 최상의 점수: 0.9736263736263737

# # reg_alambda
# da': 0, 'subsample': 1}
# 최상의 점수: 0.9736263736263737

