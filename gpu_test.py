# import numpy as np 
# import tensorflow as tf
# print(tf.__version__)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if(gpus):
#     print("쥐피유돈다")
# else:
#     print("쥐피유 안돈다 ")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ',torch.__version__, '사용 DEVICE : ', DEVICE)  

a = (112305415345)