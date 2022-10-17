from posixpath import split
import numpy as np
from regex import B
from sklearn import datasets 

a = np.array(range(1,1035))
size1 = 20 
print(a)   #[ 1  2  3  4  5  6  7  8  9 10]
###########시계열 데이터####원하는 y 값을 만들어준다 


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):  
        subset = dataset[i : (i + size)]
        aaa.append(subset)       
    return np.array(aaa)

bbb = split_x(a, size1)
print(bbb.shape)  #(1025, 10)


x = bbb[:, :-1]
y = bbb[:,  -1]
print(x)
print( y)

print(x.shape,y.shape)  #(6, 4) (6,1)
