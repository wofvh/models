
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x.shape, y.shape) #(506, 13)-> 13개의 피쳐 (506,) 
print(x_train.shape) #(16512, 8)
print(x_test.shape) #(16512, 8)

# x_train = x_train.reshape(16512, 4,2,1)
# x_test = x_test.reshape(4128, 4,2,1)
# # print(datasets.feature_names)
# print(datasets.DESCR)


#2. 모델구성
model = LinearSVR()
#3. 컴파일,훈련

model.fit(x_train,y_train)

#4. 평가,예측
results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
#ex)분류는 ACC 회귀는 R2스코어
print("results :",results)
# drop 아웃 전 
# loss : 0.32516416907310486
# r2스코어 : 0.8211444261300973
# drop 아웃 후
# oss : 0.34795650839805603
# r2스코어 : 0.7869890552305074

#CNN 

# loss : 0.3301199674606323
# r2스코어 : 0.8126296639125675

########## ML시
# results : 0.5825655411511248