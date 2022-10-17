# gpu
# criterion # 48번째 줄

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # loss 정의 방법

##### GPU #####
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch: ', torch.__version__, '사용DEVICE: ', DEVICE)

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

print(x, y)
print(x.shape, y.shape) 

# 모델구성
#model = nn.Linear(1, 1).to(DEVICE) 
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 3),
    nn.Linear(3, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1),
).to(DEVICE)           

#3. 컴파일, 훈련
criterion = nn.MSELoss() # 평가지표의 표준

#optimizer = optim.Adam(model.parameters(), lr = 0.01) # model.parameters()은 어떤 모델을 엮을것인지 즉 model = nn.Linear
optimizer = optim.SGD(model.parameters(), lr = 0.01) 

def train(model, criterion, optimizer, x, y):
    #model.train()  # 훈련모드
    optimizer.zero_grad() # 기울기 초기화
    
    hypothesis = model(x) 
    
    #loss = criterion(hypothesis, y) # 예측값과 실제값 비교 # MSE
    #loss = nn.MSELoss()(hypothesis, y) # 정상작동 (방법1)
    loss = F.mse_loss(hypothesis, y) # 정상작동 (방법2)
    
    # 여기까지가 순전파
    
    loss.backward() # 기울기값 계산까지
    optimizer.step() # 가중치 수정(역전파)
    return loss.item() 

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epoch, loss))

print("==========================================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval() # 훈련없이 평가만 하려고 함(평가모드)
    
    with torch.no_grad(): 
        predict = model(x) 
        loss2 = criterion(predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss: ', loss2)

result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값 : ', result.item())

'''
최종 loss:  0.0007705340976826847
4의 예측값 :  4.055666923522949
'''