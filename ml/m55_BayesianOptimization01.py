


param_bounds = {'x1': (-1,5),
                'x2': (0,4),}


def y_function(x1,x2):
    return -x1 **2 - (x2 - 2) **2 + 10
from pickletools import optimize
from bayes_opt import BayesianOptimization

optimize = BayesianOptimization(f = y_function, pbounds = param_bounds,
                                 random_state=1234)   #f 함수에는 파라미터를 넣음 #pbounds 파라미터 딕셔너리형태로 넣음

optimize.maximize(init_points=2, n_iter=100) #init_points=2 초기치를 잡아줌 #n_iter=5 에포크를 잡아줌 


print(optimize.max) #최대값을 출력해줌

# {'target': 9.999835918969607, 'params': {'x1': 0.00783279093916099, 'x2': 1.9898644972252864}}
