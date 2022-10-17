import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)

print(x)

# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape)  #(4, 2)

pf = PolynomialFeatures(degree=3, include_bias=False)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)  #(4, 5)

# y = wx + b

#degree=2
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

#degree=3
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]
# (4, 9)

#############################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)

print(x)

# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

print(x.shape)  #(4, 2)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)  #(4, 5)

# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]
# (4, 9)
