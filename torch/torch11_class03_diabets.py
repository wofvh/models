#logistic_regression 회기모델 (시그모이드 함수)2 진분류 0 N 1 

from calendar import EPOCH
from tkinter import Y
from unittest import result
from sklearn.datasets import load_diabetes

import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


datasets = load_diabetes()
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


# print('x_trian:',x_train.size())  
# print('x_test:',x_test.size()) 
# print('y_trian:',y_train.size())  
# print('y_test:',y_test.size()) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('########################scaler 후##################')

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size()) #torch.Size([88, 10])
print(x_train.shape)  #torch.Size([88, 10])

#2. 모델구성
class Mymodel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mymodel,self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64 ,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,8)
        self.linera5 = nn.Linear(8,output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #softmax를 안 써줘도됨 
    
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linera5(x)
        x = self.sigmoid(x)
        return x

model = Mymodel(10,1).to(DEVICE)



#3. 컴파일, 훈련
criterion = nn.MSELoss().to(DEVICE) #바이너리 크로스 엔트로피 BCE #  criterion 표준,기준

optimizer  = optim.Adam(model.parameters(), lr=0.05) # model.parameters() 모델의 가중치를 가져옴 #adam 옵티마이저 #lr 학습률


def train(model, criterion , optimizer , x_train, y_train):
    model.train() # 훈련모드로 바꿔줌 써도되고 안 써도됨 ^^
    optimizer.zero_grad()#잔여 미분값 초기화 #필수정의
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train) # criterion 표준,기준
    # y_pred = model(x_train) #모델에 x_train을 넣어서 y_pred를 예측
    # loss = criterion(y_pred, y_train) #예측값과 y_train을 비교해서 loss를 구함
    loss.backward() # 역전파를 실행하게됨 ! #필수정의
    optimizer.step()# 가중치를 갱신한다 
    return loss.item() #loss.item() 스칼라값을 반환 

EPOCHS = 1000
for epoch in range(1,EPOCHS + 1):   
    loss = train(model, criterion , optimizer , x_train, y_train)
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


# 최종 loss :  4364.61572265625
# r2_score: 0.26865163350032495