from keras.datasets import mnist, fashion_mnist , cifar10 
from tensorflow.python.keras.layers import Conv2D, MaxPool2D,Dense,Flatten ,Input,Dropout,BatchNormalizathon
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-white') 