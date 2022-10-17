from winreg import FlushKey
import numpy as np
import pandas as pd

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8, np.nan],
                     [2, 4, 6, 8 ,10],
                     [np.nan, 4, np.nan, 8,np.nan]])

print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)
print(data.shape)


#결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())



#1.결측치 삭제
print('=============결측치 삭제=================')
print(data.dropna())
print(data.dropna(axis=1))  





'''
#2-1. 특정값 - 평균 
print("===================결측치 처리 mean()=============")
means = data.mean()   #.mean 으로했을때 컬럼별 데이터 처리 (전체아님)
print('평균:',means)
data3 = data.fillna(means)
print(data3)

평균: x1    6.500000
x2    4.666667
x3    6.000000
x4    6.000000
dtype: float64
     x1        x2    x3   x4
0   2.0  2.000000   2.0  6.0
1   6.5  4.000000   4.0  4.0
2   6.0  4.666667   6.0  6.0
3   8.0  8.000000   8.0  8.0
4  10.0  4.666667  10.0  6.0

#2-1. 특정값 - 평균 
print("===================결측치 처리 mean()=============")
median = data.median()   #.mean 으로했을때 컬럼별 데이터 처리 (전체아님)
print('평균:',median)
data3 = data.fillna(median)
print(data3)

평균: x1    7.0
x2    4.0
x3    6.0
x4    6.0
dtype: float64
     x1   x2    x3   x4
0   2.0  2.0   2.0  6.0
1   7.0  4.0   4.0  4.0
2   6.0  4.0   6.0  6.0
3   8.0  8.0   8.0  8.0
4  10.0  4.0  10.0  6.0


#2-3. 특정값 - ffill, bfill  첫번째 결측치는 적용 안됨
print("==================결측치 처리 ffill, bfill==============")
data4 = data.fillna(method ='ffill')
print(data4)

     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   2.0  4.0   4.0  4.0
2   6.0  4.0   6.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0

data5 = data.fillna(method ='bfill')
print(data5)

     x1   x2    x3   x4
0   2.0  2.0   2.0  4.0
1   6.0  4.0   4.0  4.0
2   6.0  8.0   6.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN

#2-4. 특정값 - 임의값으로 채우기 ㅋ
print("==========결측치 -임의값으로 채우기==========")
data6 = data.fillna(value = 77777)
print(data6)

        x1       x2    x3       x4
0      2.0      2.0   2.0  77777.0
1  77777.0      4.0   4.0      4.0
2      6.0  77777.0   6.0  77777.0
3      8.0      8.0   8.0      8.0
4     10.0  77777.0  10.0  77777.0

print('=====================특정 칼럼만!!===================')
means =data['x1'].mean()
print(means)
data['x1'] = data['x1'].fillna(means)
print(data)

     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   6.5  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
4.0

meds = data['x2'].median()
print(meds)
data['x2'] = data['x2'].fillna(means)
print(data)

     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   6.5  4.0   4.0  4.0
2   6.0  6.5   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  6.5  10.0  NaN
     x1   x2    x3 

data['x4'] = data['x4'].fillna(77777)
print(data)

     x1   x2    x3       x4
0   2.0  2.0   2.0  77777.0
1   6.5  4.0   4.0      4.0
2   6.0  6.5   6.0  77777.0
3   8.0  8.0   8.0      8.0
4  10.0  6.5  10.0  77777.0
'''