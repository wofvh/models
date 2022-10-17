from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer #QuantileTransformer 이상치에 자유로운 놈 !
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

#1.데이터
datasets = load_iris()
x,y = datasets.data, datasets.target
print(x.shape,y.shape)  #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=1234, train_size=0.8,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#.2모델
# model = LinearRegression()
model = RandomForestRegressor()

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측    
y_predict = model.predict(x_test)
results = r2_score(y_test,y_predict)
print("구냥결과 : ", round(results,4)) 
# 기본 스코어 :  0.7665382927362877   #RF구냥결과 :  0.9196

##############################로그변환#########################################
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

print(df) #[506 rows x 13 columns]

#         CRIM    ZN  INDUS CHAS    NOX     RM   AGE     DIS  RAD    TAX PTRATIO       B LSTAT
# 0    0.00632  18.0   2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0    15.3  396.90  4.98
# 1    0.02731   0.0   7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0    17.8  396.90  9.14
# 2    0.02729   0.0   7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0    17.8  392.83  4.03
# 3    0.03237   0.0   2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0    18.7  394.63  2.94
# 4    0.06905   0.0   2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0    18.7  396.90  5.33
# ..       ...   ...    ...  ...    ...    ...   ...     ...  ...    ...     ...     ...   ...
# 501  0.06263   0.0  11.93  0.0  0.573  6.593  69.1  2.4786  1.0  273.0    21.0  391.99  9.67
# 502  0.04527   0.0  11.93  0.0  0.573  6.120  76.7  2.2875  1.0  273.0    21.0  396.90  9.08
# 503  0.06076   0.0  11.93  0.0  0.573  6.976  91.0  2.1675  1.0  273.0    21.0  396.90  5.64
# 504  0.10959   0.0  11.93  0.0  0.573  6.794  89.3  2.3889  1.0  273.0    21.0  393.45  6.48
# 505  0.04741   0.0  11.93  0.0  0.573  6.030  80.8  2.5050  1.0  273.0    21.0  396.90  7.88

# df.plot.box()
# plt.title("boston")
# plt.xlabel("Feature")
# plt.ylabel("데이터값")
# plt.show()

# print(df["B"].head())        
# df["B"] = np.log1p(df["B"]) 
# print(df["B"].head())

# df["CRIM"] = np.log1p(df["CRIM"])
# df["ZN"] = np.log1p(df["ZN"])   
# df["TAX"] = np.log1p(df["TAX"]) 

# [B,ZN, TAX]3개로  Linear로그변환 결과 :  
# ["ZN,TAX"]2개로  RandomFores로그변환 결과 :  



x_train,x_test,y_train,y_test = train_test_split(df,y,
                                                 random_state=1234, train_size=0.8,)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#.2모델
# model = LinearRegression()
model = RandomForestRegressor()

#3.훈련
model.fit(x_train,y_train)

#4.평가,예측    
y_predict = model.predict(x_test)
results = r2_score(y_test,y_predict)
print("로그변환 결과 : ", round(results,4)) 

# Linear쓰고 로그변환 결과 :  0.9278
# RandomForest쓰고 로그변환 결과 :  0.9999