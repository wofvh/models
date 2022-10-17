from logging import critical
from pickletools import optimize
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torch 
import torchvision.transforms as tr

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

path = './_data/torch_data/'

transf = tr.Compose(
    [tr.ToTensor(),
     tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = CIFAR100(path, train=True, download=False)  #train=True 학습용 데이터 #download=True 데이터를 다운로드 받겠다
test_dataset = CIFAR100(path, train=False, download=False)  #train=False 테스트용 데이터 

x_train , y_train = train_dataset.data/255., train_dataset.targets #데이터를 255로 나누어서 0~1사이의 값으로 만들어준다
x_test , y_test = test_dataset.data/255., test_dataset.targets 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

# print(x_test,y_test)

# print(np.min(x_train.numpy())), np.max((x_train.numpy())) #0.0 1.0

x_train , x_test = x_train.reshape(50000,3,32,32), x_test.reshape(10000,3,32,32) #데이터를 3차원으로 만들어준다

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle =True)#batch_size=32 한번에 32개씩 불러온다 #shuffle=True 데이터를 섞어준다
test_loader =  DataLoader(test_dset , batch_size=32, shuffle =False)
print(len(train_loader), len(test_loader)) #1563 313

#2. 모델
class CNN(nn.Module): #dropout은 test 평가할떄는 적용이 되면 안됨 훈련할때만 가능 
    def __init__(self, num_features):
        super(CNN,self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features,128, kernel_size=(3,3),stride=1),   #num_features = 784
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),                #0.5는 50%를 랜덤으로 끈다
        )
        
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(128,32, kernel_size=(3,3),),   #num_features = 784
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.3),                #0.5는 50%를 랜덤으로 끈다
        )
     
        self.hidden_layer3 = nn.Linear(32*6*6, 32)
        
        self.output_layer = nn.Linear(32,100)
        self.output_layer = nn.Linear(in_features=32,out_features=100)
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = x.view(x.shape[0], -1)     #flatten
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x
    
model = CNN(3).to(DEVICE)
# from torchsummary import summary
# summary(model, (3, 32,32))#torch summary를 사용하면 모델의 구조를 볼수있다


#3. 훈련
criterion = nn.CrossEntropyLoss().to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model,criterion,optimizer,loader):
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        
        loss = criterion(hypothesis, y_batch)
        loss.backward()#역전파
        optimizer.step()#가중치 업데이트
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, dim=1) #가장 큰값의 인덱스를 반환 
        acc = (y_predict == y_batch).float().mean() #정확도
        
        epoch_acc += acc.item()
        
    return epoch_loss/len(loader), epoch_acc/len(loader)

#hist = model.fit(x_train, y_train)   #hist 에는 loss와 acc가 들어감
#엄밀하게 말하면 hist라고 하기는 어렵고 loss와 acc가 반환해준다고함    
        
def evaluate(model, criterion,loader):
    model.eval()   #dropout은 test 평가할떄는 적용이 되면 안됨 훈련할때만 가능 "eval()"에서는 훈련이 안되기 때문에 dropout이 적용이 안된다
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad(): #no_grad 를 사용하면 gradient를 계산하지 않는다
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss = criterion(hypothesis, y_batch)
            epoch_loss += loss.item()
            
            y_predict = torch.argmax(hypothesis, dim=1)
            acc = (y_predict == y_batch).float().mean() #mean을 써서 true false 0 or 1평균을 구한다 
        
            epoch_acc += acc.item()
    return epoch_loss/len(loader), epoch_acc/len(loader)

#loss , acc = model.evaluate(x_test, y_test) #loss와 acc가 반환된다

epochs = 20
for epoch in range(1, epochs+1):
    
   loss, acc = train(model, criterion, optimizer, train_loader)
   
   val_loss, val_acc = evaluate(model, criterion, test_loader)
   
   print('epoch:{}, loss:{:.4f},acc:{:.3f},val_loss:{:.4f},val_acc:{:.3f}'\
       .format(epoch, loss, acc, val_loss, val_acc))
