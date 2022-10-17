import pandas as pd
from sklearn.datasets import load_iris,load_wine,fetch_covtype
from sklearn.cluster import KMeans   #y 값이 필요없음 
import numpy as np
from sklearn.metrics import accuracy_score, r2_score

datasets = load_wine()

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

print(df)  #[150 rows x 4 columns]


#2. 모델
kMeans = KMeans(n_clusters=3,random_state=1234) # 얼만큼 무리를 주느냐 

#.3 평가예측
kMeans.fit(df)

print(kMeans.labels_)
print(datasets.target)
'''
[0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 2 2 0 0 2 0 0 0 0 0 0 2 2
 0 0 2 2 0 0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 1 2 1 1 2 1 1 2 2 2 1 1 0
 2 1 1 1 2 1 1 2 2 1 1 1 1 1 2 2 1 1 1 1 1 2 2 1 2 1 2 1 1 1 2 1 1 1 1 2 1
 1 2 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 2 1 1 2 2 2 2 1 1 1 2 2 1 1 2 2 1 2
 2 1 1 1 1 2 2 2 1 2 2 2 1 2 1 2 2 1 2 2 2 2 1 1 2 2 2 2 2 1]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

#[실습]accuracy_score 함수를 이용해서 점수를 구하세요.

df["cluster"] = kMeans.labels_

df["target"] = datasets.target

acc = accuracy_score(df["target"], df["cluster"])
print("accuracy_score : ", acc)


# accuracy_score :  0.702247191011236
'''