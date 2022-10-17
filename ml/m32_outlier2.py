import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

aaa = np.transpose(aaa)

print(aaa)
print(aaa[:,0])
print(aaa[:,1])


def outliers(data_out,i):
    quartile_1, q2, quartile_3 = np.percentile(data_out[:,i],
                                              [25,50,75])
    print("1사분위:", quartile_1)   
    print("q2: ", q2)               
    print("3사분위:", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr:", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where(((data_out[:,i]>upper_bound)|
                    (data_out[:,i]<lower_bound)))

outliers_loc1 = outliers(aaa,0)
outliers_loc2 = outliers(aaa,1)
print("이상치의 위치:",outliers_loc1)
print("이상치의 위치:",outliers_loc2)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()
