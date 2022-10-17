#logistic_regression 회기모델 (시그모이드 함수)2 진분류 0 N 1 

from calendar import EPOCH
from tkinter import Y
from unittest import result
from sklearn.datasets import load_wine

import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


datasets = load_wine()
x = datasets.data
y = datasets.target


x = torch.FloatTensor(x)
y = torch.LongTensor(y)

print(y.unique())
# tensor([0, 1, 2])

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.2, random_state=42 ,) #stratify=y)


x_train = torch.FloatTensor(x_train)
x_test  = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train).to(DEVICE) #Float이 길어지면 Double로 바꿔줘야함
y_test = torch.LongTensor(y_test).to(DEVICE) #int가 길어지면 LONG으로 바꿔줌


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# x_test = (x_test- torch.mean(x_test))/ torch.std(x_test) # Standard Scaler
# x_train = (x_train - torch.mean(x_train))/ torch.std(x_train) # Standard Scaler

print("=============================scaler 후=============================")

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)


print(x_train.size())  #torch.Size([35, 13])
print(x_test.size())   #torch.Size([143, 13])
print(y_train.size())  #torch.Size([35,1])
print(y_train.size())  #torch.Size([35,1])

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
        self.softmax = nn.Softmax() #softmax를 안 써줘도됨 
    
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linera5(x)
        x = self.softmax(x)
        return x

model = Mymodel(13,3).to(DEVICE)


#3. 컴파일, 훈련
# criterion = nn.BCELoss().to(DEVICE) #바이너리 크로스 엔트로피 BCE #  criterion 표준,기준
criterion = nn.CrossEntropyLoss().to(DEVICE) #크로스 엔트로피 #  criterion 표준,기준 #CrossEntropyLoss 쓰면 원핫인코드을 안해줘도됨
optimizer  = optim.Adam(model.parameters(), lr=0.01) # model.parameters() 모델의 가중치를 가져옴 #adam 옵티마이저 #lr 학습률


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

EPOCHS = 100
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

loss = evaluate(model, criterion, x_test, y_test) # evaluate는 loss.item()을 반환 #
print('최종 loss : ',loss) #평가의 대한 loss는 loss 를 잡아주면 된다

# y_predict = (model(x_test) >=0.5).float() #0.5보다 크면 1, 작으면 0
# print(y_predict[:10])


y_predict = torch.argmax(model(x_test), axis=1) #argmax는 가장 큰 값의 인덱스를 반환
print(y_predict[:10])
# # y_predict = model.predict([4])

score = (y_predict == y_test).float().mean() #평균을 내서 정확도를 구함 0.5보다 크면 1, 작으면 0
print('accuracy:,{:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict) #cpu안써서 에러
# # print('accuracy_score:',(score))
# print('accuracy_score:,{:.4f}'.format(score))

score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())  # cpu로 바꿔줘야함 #np array로 바꿔줘도되고 안바꿔줘도됨
print('accuracy_score:',(score))

# accuracy:,0.9301
# accuracy_score: 0.9300699300699301