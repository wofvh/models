#mlp
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
x = np.array([range(10)])
#print(range(10))
#for i in range(10)
#   print(i)
print(x.shape)  # (1, 10)

x = np.transpose(x)
print(x.shape)  # (10, 1)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.transpose(y)   
print(y.shape)
A = np.array([[9]])

x = torch.FloatTensor(x).to(DEVICE)   # torch 타입으로 변환
y = torch.FloatTensor(y).to(DEVICE)  # 
A = torch.FloatTensor(A).to(DEVICE)  # 

A = (A - torch.mean(x))/ torch.std(x) # Standard Scaler(transform)
x = (x - torch.mean(x))/ torch.std(x) # Standard Scaler



print(A.shape) # torch.Size([1, 3])

# print(x, y) 
print(x.shape,y.shape) # torch.Size([10, 3]) torch.Size([10, 2])

#2. 모델 구성
# model = Sequential()

model = nn.Sequential(
    nn.Linear(1, 4),
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 3),
    ).to(DEVICE) # GPU 사용 시 data와 model에 무조건 wrapping하기!

#3. 컴파일, 훈련
# model.compile(loss='mse',optimizer='SGD')
criterion = nn.MSELoss() # criterion 표준,기준
optimizer = optim.Adam(model.parameters(),lr=0.01) # 모든 parameters에 맞춰 optim 적용
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
epochs = 50
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

results = model(A).to(DEVICE)


print('result : ',results.tolist())

# 최종 loss :  0.00027760813827626407
# result :  [[1.8611129522323608, 1.3490010499954224, 3.5845212936401367]]
