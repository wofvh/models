#logistic_regression 회기모델 (시그모이드 함수)2 진분류 0 N 1 

from calendar import EPOCH
from tkinter import Y
from unittest import result
from sklearn.datasets import fetch_california_housing

import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


datasets = fetch_california_housing()
x = datasets.data 
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
# print(y.unique())

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.2, random_state=42 )

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)


print('x_trian:',x_train.size())  
print('x_test:',x_test.size()) 
print('y_trian:',y_train.size())  
print('y_test:',y_test.size()) 


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('########################scaler 후##################')

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train) #torch.Size([88, 10])
print(x_train)  #torch.Size([88, 10])

###########################################################################################
from torch.utils.data import TensorDataset,DataLoader #TensorDataset는 텐서를 입력받아서 합처주는 역할을 한다.
train_set = TensorDataset(x_train, y_train) # x , y 를 합친다 
test_set = TensorDataset(x_test, y_test) # x , y 를 합친다

print(train_set)
print('-===============train_set[0]=================================')
print(train_set[0])
print('-===============train_set[0][0]=================================')
print(train_set[0][0])
print('-===============train_set[0][1]=================================')
print(train_set[0][1])
print('-===============train_setlen=================================')
print(len(train_set))  #309
#x.y 배치를 합체한다 
train_loader  = DataLoader(train_set, batch_size=40, shuffle=True) #DataLoader은 데이터를 미니배치 단위로 끊어서 가져올 수 있게 해준다.
test_loader  = DataLoader(test_set, batch_size=40, shuffle=False)

#2. 모델구성
class Mymodel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mymodel,self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64 ,32)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(32,16)
        self.linera4 = nn.Linear(16,output_dim)
    
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.linera4(x)
        return x

model = Mymodel(8,1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss().to(DEVICE) #바이너리 크로스 엔트로피 BCE #  criterion 표준,기준

optimizer  = optim.Adam(model.parameters(), lr=0.05) # model.parameters() 모델의 가중치를 가져옴 #adam 옵티마이저 #lr 학습률


def train(model, criterion , optimizer , loader):
    model.train() # 훈련모드로 바꿔줌 써도되고 안 써도됨 ^^
    total_loss = 0
    
    for x_bacth, y_bacth in loader:
        optimizer.zero_grad()#잔여 미분값 초기화 #필수정의
        hypothesis = model(x_bacth)
        loss = criterion(hypothesis, y_bacth) # criterion 표준,기준
    
        loss.backward() # 역전파를 실행하게됨 ! #필수정의
        optimizer.step()# 가중치를 갱신한다 
        total_loss += loss.item()
    return total_loss/len(loader) 

EPOCHS = 1000
for epoch in range(1,EPOCHS + 1):   
    loss = train(model, criterion , optimizer , train_loader)
    if epoch % 20==0:
        print('epoch {}, loss: {:.8f}'.format(epoch, loss)) 


#4. 평가, 예측
print('======================평가, 예측======================')

def evaluate(model, criterion,loader): #평가할 때는 test는 미분을 하지 않음 
    model.eval()    #eval 은 무조건 명시해줘야함
    total_loss = 0

    for x_bacth, y_bacth in loader:
        with torch.no_grad():
            y_predict = model(x_bacth)
            loss = criterion(y_predict, y_bacth)
            total_loss = loss.item()
        return total_loss

loss = evaluate(model, criterion, test_loader) # evaluate는 loss.item()을 반환
print('최종 loss : ',loss) #평가의 대한 loss는 loss 를 잡아주면 된다

y_predict = model(x_test)

from sklearn.metrics import r2_score

score = r2_score(y_test.detach().cpu().numpy(), y_predict.detach().cpu().numpy())  # cpu로 바꿔줘야함 #np array로 바꿔줘도되고 안바꿔줘도됨
print('r2_score:',(score))


# 최종 loss :  0.3241405189037323
# r2_score: 0.7039949363999811
