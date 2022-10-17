# pytorch titcianic
from calendar import EPOCH
from tkinter import Y
from unittest import result
from sklearn.datasets import load_digits
import pandas as pd
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)


path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path +'train.csv')

test_set = pd.read_csv(path + 'test.csv',index_col=0) 


print(train_set.Pclass.value_counts()) 

Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")


female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")

sns.barplot(x="SibSp", y="Survived", data=train_set)

       
# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)

# print(test_set.columns)
# print(train_set.info()) # info 정보출력
# print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 1. 제거 ####

train_set = train_set.fillna({"Embarked": "S"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

############################



x = train_set.drop(['Survived', 'PassengerId'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived'] 
print(y)
print(y.shape) # (891,)

x = torch.FloatTensor(x.values)
y = torch.FloatTensor(y.values)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66,
                                                    shuffle=True)
                                                    

print(x_train.size(), x_test.size(), y_train.size(), y_test.size()) 

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

print(x_train.size(), x_test.size(), y_train.size(), y_test.size()) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('########################scaler 후##################')

print('x_trian:',x_train)  
print('x_test:',x_test) 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size()) #torch.Size([623, 8])
print(x_train.shape)  #torch.Size([623, 8])

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
        self.linear3 = nn.Linear(32,16)
        self.linera4 = nn.Linear(16,output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.linera4(x)
        x = self.sigmoid(x)
        return x

model = Mymodel(8,1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss().to(DEVICE) #바이너리 크로스 엔트로피 BCE #  criterion 표준,기준

optimizer  = optim.Adam(model.parameters(), lr=0.01) # model.parameters() 모델의 가중치를 가져옴 #adam 옵티마이저 #lr 학습률


def train(model, criterion , optimizer ,loader):
    model.train() # 훈련모드로 바꿔줌 써도되고 안 써도됨 ^^
    total_loss = 0
    
    for x_bacth, y_bacth in loader:
        optimizer.zero_grad()#잔여 미분값 초기화 #필수정의
        hypothesis = model(x_bacth)
        loss = criterion(hypothesis, y_bacth) # criterion 표준,기준
     
        loss.backward() # 역전파를 실행하게됨 ! #필수정의
        optimizer.step()# 가중치를 갱신한다 
        total_loss += loss.item()
    return total_loss/len (loader) #loss.item() 스칼라값을 반환 

EPOCHS = 100
for epoch in range(1,EPOCHS + 1):   
    loss = train(model, criterion , optimizer , train_loader)
    if epoch % 20 == 0:
        print('epoch {}, loss: {:.8f}'.format(epoch, loss)) 


#4. 평가, 예측
print('======================평가, 예측======================')

def evaluate(model, criterion, loader): #평가할 때는 test는 미분을 하지 않음 
    model.eval()    #eval 은 무조건 명시해줘야함
    total_loss = 0

    for x_bacth , y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_bacth)
            loss = criterion(y_predict, y_batch)
            total_loss = loss.item()
        return total_loss


loss = evaluate(model, criterion, test_loader) # evaluate는 loss.item()을 반환
print('최종 loss : ',loss) #평가의 대한 loss는 loss 를 잡아주면 된다

y_predict = (model(x_test) >=0.5).float() #0.5보다 크면 1, 작으면 0
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


# accuracy:,0.7985
# accuracy_score: 0.7985074626865671