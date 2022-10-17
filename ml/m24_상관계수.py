import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터

datasets = load_iris()

print(datasets.feature_names)

x = datasets.data
y = datasets.target

df = pd.DataFrame (x, columns=[['sepal length', 'sepal width', 'petal length', 'petal width']])

df['target(y)'] = y
print(df)


print("============== 상관계수 히트 맵 ===================")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot =True, cbar =True)
plt.show()

