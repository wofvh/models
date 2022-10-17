import numpy as np


aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
aaa = aaa.reshape(-1,1) # 행렬로 변환하기 위해 reshape()함수를 사용함
print(aaa.shape) # (13, 1)
print(aaa) # (13, 1)

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope(contamination=0.5) # contamination

outlier.fit(aaa)
results = outlier.predict(aaa)
print(results) # [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]


# [[-10]
#  [  2]

#  [  3]
#  [  4]
#  [  5]
#  [  6]
#  [  7]
#  [  8]
#  [  9]
#  [ 10]
#  [ 11]
#  [ 12]
#  [ 50]]
# [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]