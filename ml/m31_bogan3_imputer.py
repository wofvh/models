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


'''
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer, IterativeImputer
# imputer = SimpleImputer(strategy='mean')    
# imputer = SimpleImputer(strategy='median')    #simpleimputer 평균값으로 nan 값처리
# imputer = SimpleImputer(strategy='most_frequent')  #most_frequent nan값에 
# imputer = SimpleImputer(strategy='constant')  #constant nan값에0 이들어감  
# imputer = SimpleImputer(strategy='constant',fill_value=7777)  #fill_value nan 값에  정해진 숫자가 들어감 

# imputer = IterativeImputer()
imputer = KNNImputer()

 
imputer.fit(data)
data2 = imputer.transform(data)
print(data2)
'''