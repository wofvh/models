from re import A
import numpy as  np
import torch
print(torch.__version__) # 1.12.1

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch :',torch.__version__,'\n','사용DEVICE : ',DEVICE)
print(torch.cuda.device_count())


#1. 데이터
x  = np.array([1, 2, 3])    # (3,)
y  = np.array([1, 2, 3])    # (3,)

# torch는 numpy 형태가 아닌 torch tensor로 변환해줘야한다.
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)   # (3,) => # (3, 1)
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)  # (3,) => # (3, 1)

A = (4 - torch.mean(x))/ torch.std(x) # Standard Scaler

x = (x - torch.mean(x))/ torch.std(x) # Standard Scaler
print(A)




print(x, y) 
print(x.shape,y.shape) # torch.Size([3, 1]) torch.Size([3, 1])


#2. 모델 구성
# model = Sequential()
model = nn.Linear(1, 1).to(DEVICE) # 인풋 x의 컬럼 / 아웃풋 y의 컬럼

#3. 컴파일, 훈련
# model.compile(loss='mse',optimizer='SGD')
criterion = nn.MSELoss() # criterion 표준,기준
optimizer = optim.SGD(model.parameters(),lr=0.01) # 모든 parameters에 맞춰 optim 적용
# optim.Adam(model.parameters(),lr=0.01) # 모든 parameters에 맞춰 optim 적용


def train(model, criterion, optimizer, x, y ):
    # model.train()         # 훈련 mode (디폴트라서 명시 안하면 train mode임)
    optimizer.zero_grad()   # 1.손실함수의 기울기를 초기화
    hypthesis = model(x)
    # loss =  nn.MSELoss(hypthesis, y) # 에러
    # loss =  nn.MSELoss()(hypthesis, y)
    loss  = criterion(hypthesis,y)
    
    loss.backward()         # 2.가중치 역전파
    optimizer.step()        # 3.가중치 갱신
    return loss.item()
epochs = 2000
for epoch in range(1, epochs +1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss : {}'.format(epoch,loss))

#4. 평가, 예측
# loss = model.evaluate(x_test,y_test)
def evaluate(model, criterion, x, y):
    model.eval()            # 평가 mode

    with torch.no_grad():
        y_predict = model(x)
        results = criterion(y_predict,y)
    return results.item()


loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ',loss2)

# y_predict = model.predict([4])

results = model(torch.Tensor([A]).to(DEVICE))


print('result : ',results.item())

# 최종 loss :  3.7203865304
# result :  4.003868579

