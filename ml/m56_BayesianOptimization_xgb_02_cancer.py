from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier,XGBRFRegressor

#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (569, 30) (569,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123  
)


Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델

Bayesian_parameters = {
    'max_depth' : (6, 16),
    #'num_leaves' : (24, 64),
    #'min_child_samples' : (10,200),
    # 'gamma' :(1, 2),
    'min_child_weight' : (1,50),
    'subsample' : (0.5,1),
    'colsample_bytree' : (0.5,1),
    'max_bin' : (10,500),
    'reg_lambda' : (0.001,10),
    'reg_alpha' : (0.01,50),
    "learning_rate" : (0.1,1.5),
}

# {'target': 0.9241285771962268, 
# 'params': {'colsample_bytree': 1.0, 
#            'max_bin': 110.44476552296565, 
#            'max_depth': 16.0, 
#            'min_child_samples': 10.0, 
#             'min_child_weight': 1.0, 
#             'num_leaves': 64.0,     
#             'reg_alpha': 0.01, 
#             'reg_lambda': 10.0, 
#             'subsample': 1.0}}

def lgb_hamus(max_depth, subsample,min_child_weight, colsample_bytree, max_bin, reg_lambda, reg_alpha,learning_rate):
    params = {
        
        'learning_rate': int(round(learning_rate)),
        'max_depth' : int(round(max_depth)),  #round 반올림 할때 사용  무조건 정수로 바꿔줘야함
        # 'gamma': int(round(gamma)),
        #'min_child_samples': int(round(min_child_samples)),
        #'min_child_weight': int(round(min_child_weight)),  #round 반올림 할때 사용 무조건 정수로 바꿔줘야함
        'min_child_weight': int(round(min_child_weight)),
        'subsample': max(min(subsample,1),0),   #subsample은 0~1 사이의 값만 받아드림
        'colsample_bytree': max(min(colsample_bytree,1),0),    #colsample_bytree는 0~1 사이의 값만 받아드림
        'max_bin': max(int(round(max_bin)),10), #max_bin은 10이상의 값만 받아드림 정수형으로 받아드림
        'reg_lambda': max(reg_lambda,0),    #reg_lambda는 0이상의 값만 받아드림 양수형으로 받아드림
        'reg_alpha': max(reg_alpha,0), #reg_alpha는 0이상의 값만 받아드림 양수형으로 받아드림
    }
    
    #*여러개의 인자를 받겠다는 의미
    #**키워드를 {딕셔너리형태로 받겠다}
    
    model = XGBClassifier(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='mlogloss',
              verbose=0,
              early_stopping_rounds=50)
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    print('최종 점수:',results)
    return results  

# lgb_bo = BayesianOptimization(f=lgb_hamus,
#                                     pbounds = Bayesian_parameters,
#                                     random_state=123)

# lgb_bo.maximize(init_points=5, n_iter=50)
# print(lgb_bo.max)

# from xgboost import XGBClassifier,XGBRegressor
# from sklearn.model_selection import GridSearchCV

# model = XGBRegressor({'n_estimators': 500, "learning_rate": 0.02,
#                       'learning_rate': int(round(0.5005950689305312)),
#                       'max_depth' : int(round(11.513147690828912,)),  #round 반올림 할때 사용  무조건 정수로 바꿔줘야함
#                       'subsample': max(min(0.9903820991923078,1),0),   #subsample은 0~1 사이의 값만 받아드림
#                       'colsample_bytree': max(min(0.8482345927989308, 1),0),    #colsample_bytree는 0~1 사이의 값만 받아드림
#                       'max_bin': max(int(round(121.15721224645952,)),10), #max_bin은 10이상의 값만 받아드림 정수형으로 받아드림
#                       'reg_lambda': max(4.231641494784486,0),    #reg_lambda는 0이상의 값만 받아드림 양수형으로 받아드림
#                       'reg_alpha': max(35.9762537995803,0), #reg_alpha는 0이상의 값만 받아드림 양수형으로 받아드림
#                       })

# #3.훈련
# import time
# start = time.time()
# model.fit(x_train, y_train)
# end = time.time()

# #4.결과
# results = model.score(x_test, y_test)
# print('결과:',results)
# print('걸린시간:',end - start)
