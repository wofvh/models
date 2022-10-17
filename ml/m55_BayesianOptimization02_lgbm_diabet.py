from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123  
)


Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델

Bayesian_parameters = {
    'max_depth' : (6, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10,200),
    'min_child_weight' : (1,50),
    'subsample' : (0.5,1),
    'colsample_bytree' : (0.5,1),
    'max_bin' : (10,500),
    'reg_lambda' : (0.001,10),
    'reg_alpha' : (0.01,50),
}


def lgb_hamus(max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators': 500,"learning_rate": 0.02,
        'max_depth' : int(round(max_depth)),  #round 반올림 할때 사용  무조건 정수로 바꿔줘야함
        'num_leaves': int(round(num_leaves)),
        'min_child_samples': int(round(min_child_samples)),
        'min_child_weight': int(round(min_child_weight)),  #round 반올림 할때 사용 무조건 정수로 바꿔줘야함
        'subsample': max(min(subsample,1),0),   #subsample은 0~1 사이의 값만 받아드림
        'colsample_bytree': max(min(colsample_bytree,1),0),    #colsample_bytree는 0~1 사이의 값만 받아드림
        'max_bin': max(int(round(max_bin)),10), #max_bin은 10이상의 값만 받아드림 정수형으로 받아드림
        'reg_lambda': max(reg_lambda,0),    #reg_lambda는 0이상의 값만 받아드림 양수형으로 받아드림
        'reg_alpha': max(reg_alpha,0), #reg_alpha는 0이상의 값만 받아드림 양수형으로 받아드림
    }
    #*여러개의 인자를 받겠다는 의미
    #**키워드를 {딕셔너리형태로 받겠다}
    model = LGBMRegressor(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    
    return results  

lgb_bo = BayesianOptimization(f=lgb_hamus,
                                    pbounds = Bayesian_parameters,
                                    random_state=123)

lgb_bo.maximize(init_points=5, n_iter=100)
print(lgb_bo.max)

# {'target': 0.6213997653299669, 'params': {'colsample_bytree': 0.5875493964947438, 'max_bin': 147.0269960297086,
#                                           'max_depth': 14.759415192212911, 'min_child_samples': 32.033875714919915,
#                                           'min_child_weight': 15.209850292243916, 'num_leaves': 45.57513689706032,
#                                           'reg_alpha': 27.624495286092777, 'reg_lambda': 5.989493688096531,
#                                           'subsample': 0.6122818424641512}}  


####################[실습]###############################
#1. 수정한 파라미터로 모델 만들어서 비교!!!
#2. 수정한 파라미터를 이용해서 

# {'target': 0.6213997653299669,
#  'params': {'colsample_bytree': 0.5875493964947438, 
# 'max_bin': 147.0269960297086, 'max_depth': 14.759415192212911,
# 'min_child_samples': 32.033875714919915, 'min_child_weight': 15.209850292243916, 
# 'num_leaves': 45.57513689706032, 'reg_alpha': 27.624495286092777, 'reg_lambda': 5.989493688096531,
# 'subsample': 0.6122818424641512}}
