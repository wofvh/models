from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_boston,load_wine
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
datasets = load_wine()
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


# df.plot.box()
# plt.title("boston")
# plt.xlabel("Feature")
# plt.ylabel("데이터값")
# plt.show()

# print(df["B"].head())         #Linear 기본 스코어 :  0.7665382927362877 
# df["B"] = np.log1p(df["B"])  # Linear log 변환후! 구냥결과 :  0.7711
# print(df["B"].head())

# df["CRIM"] = np.log1p(df["CRIM"])#Linear로그변환 결과 :  0.7596
# df["ZN"] = np.log1p(df["ZN"])    #Linear로그변환 결과 :  0.7734
# df["TAX"] = np.log1p(df["TAX"]) #Linear로그변환 결과 :  0.7669

#[B,ZN, TAX]3개로  Linear로그변환 결과 :  0.7785 제일 좋았다~!     Linear 기본 스코어 :  0.766
#[ "ZN,TAX"]2개로  RandomFores로그변환 결과 :  0.92 제일 좋았다~! "]  RF구냥결과 :  0.9196



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
# Linear 기본 스코어 :  0.7665382927362877   #RF구냥결과 :  0.9196

# Linear log 변환후! 구냥결과 :  0.7711
