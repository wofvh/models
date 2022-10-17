import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

x = np.array([1,2,3])
y = np.array([1,2,3])

x  = torch.FloatTensor(x).unsqueeze(1)
y = torch.FloatTensor(y).unsqueeze(-1)

print(x,y)
print(x.shape,y.shape)

#2. 모델구성
# model = Sequential()
model = nn.Linear(1,1)#input x, output y / 단층레이어, 선형회귀

#3. 컴파일 훈련
criterion  = nn.MSELoss()# 로스는 표준 , mse
optimizer = optim.SGD(model.parameters(), lr=0.01)#Parameter를 모두 업데이트 하겠다. model.parameters() / lr=0.01 <<무조건 명시해야함

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad() #돌면서 미분해서 남은 값을을 0 으로 초기화#역전파 전에 gradient를 0으로 초기화
    
    hypothesis = model(x) #모델에 x를 넣어서 예측값을 도출 #y = wx + b 손실함수 계산을 위해 예측값을 도출
    
    loss = criterion(hypothesis, y) #예측값과 실제값을 비교해서 손실값을 도출 #손실값을 계산
    
    loss.backward()     #손실값을 기준으로 역전파를 수행 #역전파를 수행 #손실값을 기준으로 역전파를 수행
    optimizer.step()    #업데이트 #업데이트
    return loss.item()  #loss.item()은 loss의 값을 스칼라 값으로 도출 

epochs = 500
for epoch in range(1 , epochs +1):
    loss = train(model, criterion , optimizer, x, y)
    print("epoch : {}, loss : {}".format(epoch, loss)) 
    
'''
파이토치 반복 문법
    1.optimizer.zero_grad()   #손실함수 기울기 초기화
    2.loss.backward() 역전파
    3.optimizer.step 역전파를 하면서 웨이트 갱신
    1 epoch = 1-> 2-> 3

'''
def evaluate(model , criterion , x,y):
    model.eval() #모델을 평가 모드로 전환 #평가모드 (반드시 명시해야함)
    
    with torch.no_grad():
        x_predict = model(x)
        results = criterion(x_predict, y)
    return results.item()

loss2 = evaluate(model, criterion, x,y)
print("최종 loss :", loss2)

results = model(torch.tensor([[4.0]])) #[[4.0]] 2차원으로 만들어줘야함 예) [[[]]] 3차원
print('4의 예측값:', results.item())

