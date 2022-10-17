from mmap import mmap
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,TensorDataset #데이터를 합쳐주는 역할을한다 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torchvision.transforms as tr

USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

transf = tr.Compose([tr.Resize(150),tr.ToTensor()]) #평균과 표준편차를 정규화한다
path = './_data/torch_data/'

# train_dataset = MNIST(path, train=True, download=False,transform=transf)  #train=True 학습용 데이터 #download=True 데이터를 다운로드 받겠다
# test_dataset = MNIST(path, train=False, download=False,transform=transf)  #train=False 테스트용 데이터 
train_dataset = MNIST(path, train=True, download=False)
test_dataset = MNIST(path, train=False, download=False)

# print(train_dataset[0][0].size()) #torch.Size([1, 15, 15])


x_train , y_train = train_dataset.data/255., train_dataset.targets #데이터를 255로 나누어서 0~1사이의 값으로 만들어준다
x_test , y_test = test_dataset.data/255., test_dataset.targets 

print(x_train.shape ,x_test.size())  #torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
print(y_train.shape ,y_test.size())  #torch.Size([60000]) torch.Size([10000])


print(np.min(x_train.numpy())), np.max((x_train.numpy())) #0.0 1.0

#텐서플로우랑 파이토치가 다른점 
#60000,28,28 -> 60000,1,28,28

# x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784) #28*28 = 784 #view는 차원을 바꿔준다(reshape와 같은 역할)

x_train , x_test = x_train.unsqueeze(1), x_test.unsqueeze(1) #1차원을 추가해준다

print(x_train.shape ,x_test.size())  #torch.Size([60000, 784]) torch.Size([10000, 784]) #cnn은 4차원을 받는다 [660000,1,28,28]

train_dset = TensorDataset(x_train, y_train) # x , y 를 합친다 #스케일링도 합침
test_dset  = TensorDataset(x_test, y_test) # x , y 를 합친다 #TensorDataset은 클레스로 정의되어있다 

train_loader = DataLoader(train_dset, batch_size=32, shuffle =True)#batch_size=32 한번에 32개씩 불러온다 #shuffle=True 데이터를 섞어준다
test_loader =  DataLoader(test_dset , batch_size=32, shuffle =False)


#2. 모델
class CNN(nn.Module): #dropout은 test 평가할떄는 적용이 되면 안됨 훈련할때만 가능 
    def __init__(self, num_features):
        super(CNN,self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features,64, kernel_size=(3,3),stride=1),   #num_features = 784
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.3),                #0.5는 50%를 랜덤으로 끈다
        )
        
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=(3,3),),   #num_features = 784
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.3),                #0.5는 50%를 랜덤으로 끈다
        )
        self.hidden_layer3 = nn.Linear(32*5*5,32)
        
        # self.flatten = nn.Flatten()
            
        # self.hidden_layer5 = nn.Sequential(
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),  
        # )
        self.output_layer = nn.Linear(in_features=32,out_features=10)
        
    def forward(self,x): #nn 모듈을 상속받았기 때문에 forward를 사용해야한다
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = x.view(x.shape[0], -1)     #flatten
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x
    
model = CNN(1).to(DEVICE)


#3. 훈련,컴파일
criterion = nn.CrossEntropyLoss().to(DEVICE) #손실함수

optimizer = optim.Adam(model.parameters(), lr=1e-4) #0.0001 #최적화 함수 1이 있는곳까지센다

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

