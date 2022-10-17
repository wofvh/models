#logistic_regression 회기모델 (시그모이드 함수)2 진분류 0 N 1 
from calendar import EPOCH
from sklearn.datasets import fetch_california_housing
import pandas as pd
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm_notebook
import matplotlib
from sklearn.preprocessing import LabelEncoder

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
# submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
#                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


print(test_set)
print(train_set.shape) #(1460,76)
print(test_set.shape) #(1459, 75) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']

x = torch.FloatTensor(x.values)
y = torch.FloatTensor(y.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.2, random_state=42 )


x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

print(x_train.size())   #torch.Size([292, 75])
print(x_test.size())    #torch.Size([1168, 75])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print('============================scaler후 ==========================')

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())   #torch.Size([292, 75])
print(x_test.size())    #torch.Size([1168, 75])

#2. 모델 
class Mymodel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mymodel,self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128,64)
        self.sigmoid = nn.Sigmoid()
        self.linear3 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear4 = nn.Linear(32 ,8)
        self.linear5 = nn.Linear(8, output_dim)
    def forward(self,input_size):
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x
        
model = Mymodel(75,1).to(DEVICE)

#3. 컴파일 훈련

criterion = nn.MSELoss().to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.10)

def train(model, criterion,optimizer, x_train ,y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 1000
for epoch in range(EPOCHS+1):
    loss = train(model, criterion , optimizer,x_train ,y_train)
    print('epoch {}, loss: {:.8f}'.format(epoch, loss)) 
    


#4. 평가, 예측
print('======================평가, 예측======================')

def evaluate(model, criterion, x_test, y_test): #평가할 때는 test는 미분을 하지 않음 
    model.eval()    #eval 은 무조건 명시해줘야함

    with torch.no_grad():
        y_predict = model(x_test)
        loss = criterion(y_predict, y_test)
    return loss.item()


loss = evaluate(model, criterion, x_test, y_test) # evaluate는 loss.item()을 반환
print('최종 loss : ',loss) #평가의 대한 loss는 loss 를 잡아주면 된다
y_predict = model(x_test)


from sklearn.metrics import r2_score

score = r2_score(y_test.detach().cpu().numpy(), y_predict.detach().cpu().numpy())  # cpu로 바꿔줘야함 #np array로 바꿔줘도되고 안바꿔줘도됨
print('r2_score:',(score))


# 최종 loss :  2079226496.0
# r2_score: 0.6838466825739029