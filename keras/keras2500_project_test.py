from matplotlib.pyplot import hist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.model_selection import train_test_split


season = ImageDataGenerator(
    rescale=1./255)

season1 = season.flow_from_directory(
    'C:/study/_data/hoho/',
    target_size=(70,70),# 크기들을 일정하게 맞춰준다.
    batch_size=50,
    class_mode='categorical', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )

np.save('d:/study_data/_save/_npy/face_project13.npy',  arr=season1[0][0])