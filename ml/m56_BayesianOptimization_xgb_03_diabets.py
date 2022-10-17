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
print(x.shape, y.shape) # (506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123  
)


Scaler = StandardScaler() #Bagging 할때 스케일러 필수 
x_train = Scaler.fit_transform(x_train)
x_test = Scaler. transform(x_test)

#.2 모델
bayesian_params = {
    'max_depth' : [2,10], #default 3/ 0~inf(무한대) / 정수 => 소수점은 정수로 변환하여 적용해야 함
    'gamma': [45,50], #default 0 / 0~inf
    'min_child_weight': [27,30], #default 1 / 0~inf
    'subsample' : [0.9,1], #default 1 / 0~1
    'colsample_bytree' : [0.5,1], #default 1 / 0~1
    'colsample_bylevel' : [0.8,1], #default 1 / 0~1
    'colsample_bynode' : [0.9,1], #default 1 / 0~1
    'reg_alpha' : [26,30], #default 0 / 0~inf / L1 절대값 가중치 규제 / 그냥 alpha도 적용됨
    'reg_lambda' : [2,100], #default 1 / 0~inf / L2 제곱 가중치 규제 / 그냥 lambda도 적용됨
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


def lgb_hamus(max_depth, gamma, min_child_weight,  
              subsample, colsample_bytree, colsample_bylevel, colsample_bynode, 
              reg_lambda, reg_alpha):
    params = {
     'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                # 무조건 정수
        'gamma' : int(round(gamma)), 
        'min_child_weight' : int(round(min_child_weight)),  
        'subsample' : max(min(subsample, 1), 0),             # 0~1 사이의 값
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),   
        'colsample_bylevel' : max(min(colsample_bylevel, 1), 0),   
        'colsample_bynode' : max(min(colsample_bynode, 1), 0),   
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : max(reg_alpha, 0),
    }
    #*여러개의 인자를 받겠다는 의미
    #**키워드를 {딕셔너리형태로 받겠다}
    model = XGBRFRegressor(**params,verbose=False)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    
    return results  

lgb_bo = BayesianOptimization(f=lgb_hamus,
                                    pbounds = bayesian_params,
                                    random_state=123)

lgb_bo.maximize(init_points=5, n_iter=50)
print(lgb_bo.max)
