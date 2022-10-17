#logistic_regression 회기모델 (시그모이드 함수)2 진분류 0 N 1 
from calendar import EPOCH
from sklearn.datasets import fetch_california_housing
import pandas as pd
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

path = './_data/ddarung/'
train_set = pd.read_csv(path +'train.csv')

test_set = pd.read_csv(path + 'test.csv',index_col=0) 
 
print(test_set)
print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set['count']

x = torch.FloatTensor(x.values)
y = torch.FloatTensor(y.values)
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

#2. 모델구성
model  = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.Sigmoid(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
).to(DEVICE)


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

# def evaluate(model, criterion,x,y):
#     model.eval()
    
#     with torch.no_grad(): #평가할 때는 미분을 하지 않는다
#         y_predict = model(x_test)
#         results = criterion(y_predict, y_test)
#     return results.item()

# loss2 = evaluate(model, criterion, x_test, y_test)
# print('loss2:',loss2)

# results = model(x_test).to(DEVICE)
# print('results:',results)

def evaluate(model, criterion, x_test, y_test): #평가할 때는 test는 미분을 하지 않음 
    model.eval()    #eval 은 무조건 명시해줘야함

    with torch.no_grad():
        y_predict = model(x_test)
        loss = criterion(y_predict, y_test)
    return loss.item()


loss = evaluate(model, criterion, x_test, y_test) # evaluate는 loss.item()을 반환
print('최종 loss : ',loss) #평가의 대한 loss는 loss 를 잡아주면 된다

# y_predict = (model(x_test) >=0.5).float() #0.5보다 크면 1, 작으면 0
# print(y_predict[:10])
# # y_predict = model.predict([4])
y_predict = model(x_test)

# score = (y_predict == y_test).float().mean() #평균을 내서 정확도를 구함 0.5보다 크면 1, 작으면 0
# print('r2_score:,{:.4f}'.format(score))

from sklearn.metrics import r2_score
# score = accuracy_score(y_test, y_predict) #cpu안써서 에러
# # print('accuracy_score:',(score))
# print('accuracy_score:,{:.4f}'.format(score))

score = r2_score(y_test.detach().cpu().numpy(), y_predict.detach().cpu().numpy())  # cpu로 바꿔줘야함 #np array로 바꿔줘도되고 안바꿔줘도됨
print('r2_score:',(score))


# 최종 loss :  2689.358154296875
# r2_score: 0.6052634231139864