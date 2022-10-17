import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)
print(aaa.shape) # (13, 2)
aaa1 = aaa[:,0].reshape(-1,1) # (13, 1)
aaa2 = aaa[:,1].reshape(-1,1)
print(aaa1) # (13,)
print(aaa2) # (13,)
print(aaa1.shape) # (13,)
print(aaa2.shape) # (13,)

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope(contamination=0.1) # contamination

outlier1 = outlier.fit(aaa1)
results1 = outlier1.predict(aaa1)
print(results1) # [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]

outlier2 = outlier.fit(aaa2)
results2 = outlier2.predict(aaa2)
print(results2) # [ 1  1  1  1  1  1 -1  1 -1 -1  1  1  1]