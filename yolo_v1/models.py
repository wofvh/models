# yolo v1
import torch
import numpy as np
import torch.nn as nn


architecture_config = [
    (7, 64, 2, 3),  # kernel_size, filters, stride, padding
    "M",  # maxpooling
    (3, 192, 1, 1),# kernel_size, filters, stride, padding
    "M",  # maxpooling
    (1, 128, 1, 0),# kernel_size, filters, stride, padding 
    (3, 256, 1, 1),# kernel_size, filters, stride, padding
    (1, 256, 1, 0),# kernel_size, filters, stride, padding
    (3, 512, 1, 1),# kernel_size, filters, stride, padding
    "M",  # maxpooling
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],# kernel_size, filters, stride, padding # 4 반복횟수
    (1, 512, 1, 0),# kernel_size, filters, stride, padding
    (3, 1024, 1, 1),# kernel_size, filters, stride, padding
    "M",  # maxpooling
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],# kernel_size, filters, stride, padding # 2번 반복
    (3, 1024, 1, 1),# kernel_size, filters, stride, padding
    (3, 1024, 2, 1),# kernel_size, filters, stride, padding
    (3, 1024, 1, 1),# kernel_size, filters, stride, padding
    (3, 1024, 1, 1),# kernel_size, filters, stride, padding
] 


class CNNBlock(nn.Module): #클래스 정의 이유는 
    def __init__(self, in_channels, out_channels, **kwaras):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwaras) #bias = False 배치놈 쓸거니까
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1) # CNNBlock(3, 32, kernel_size=3, stride=1, padding=1) # 3은 채널수 32는 필터수 #LeakyReLU(0.1) 0.1은 기울기
    
    def forward(self, x):
         return self.leakyrelu(self.batchnorm(self.conv(x))) #

class YOLOv1(nn.Module): #darnknet
    def __init__(self, in_channels=3, **kwargs): 
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config, #architecture_config는 위에있는 리스트
        self.in_channels = in_channels#channel = 3
        self.darknet = self._create_conv_layers(self.architecture)  #darnknet joseph redmon이 만들었다고 함
        self.fcs = self._create_fcs(**kwargs) # 1번째 conv2d #fcs = fully connected layer

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1)) #dim = 1 이면 1번째 차원을 기준으로 펼친다. 

    def _create_conv_layers(self, architecture): #convolution layer
        layers = [] # layer 안에  cnnblock이 들어갈거다
        in_channels = self.in_channels # in_channels = 3

        for x in architecture: #architecture_config 리스트를 x에 넣어서 반복
            if type(x) == tuple: # x가 튜플이면 cnnblock을 추가한다
                layers += [
                    CNNBlock(
                        in_channels , x[1], kernel_size=x[0], strd=x[2], Padding =x [3],
                    )
                ] 
                in_channels = x[1] # in_channels = x[1] # in_channels = 64

                                
            elif type(x) == str: # x가 스트링이면 
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0] # conv1 = (1, 256, 1, 0) #tuple 
                conv2 = x[1] # conv2 = (3, 512, 1, 1) #tuple 
                num_repeatrs = x[2] # 반복횟수는 x 2번째에 # intehger  #모든 리스트는 2개의 convd 레이어가 있음
                
                for _ in range(num_repeatrs):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1], 
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3], #cnnblock에서 호출
                    )
                ]
                    layers += [
                        CNNBlock( 
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                    )
                ]
                in_channels = conv2[1] # in_channels = 512 두번째 conv2[1] = 512 #두번쨰 요소의 레이어의 출력채널수를 in_channels로 설정

        return nn.Sequential(*layers) # *layers = 리스트를 언패킹해서 넣어준다.
       

    def _create_fcs(self, split_size, num_boxes, num_classes): #fully connected layer #split_size = 7 #num_boxes = 2 #num_classes = 20
        S, B, C = split_size, num_boxes, num_classes # S = 7 # B = 2 # C = 20 # S = split_size # B = num_boxes # C = num_classes 분할크기 박스개수 클래스개수
        return nn.Sequential(
            nn.Flatten(),# 1차원으로 펼친다
            nn.Linear(1024 * S * S, 496), # 1024 * 7 * 7 = 50176 # 496 = 1024 * 7 * 7 / 2 # 1024는 3번째 conv2d의 필터수 # 7 * 7은 split_size #
            nn.Dropout(0.0), # 0.0 = 0% # 0.5 = 50%    #기본적으로 0.5
            nn.LeakyReLU(0.1), # LeakyReLU(0.1) 0.1은 기울기
            nn.Linear(496, S * S * (C + B * 5)), # 496 = 1024 * 7 * 7 / 2 # 7 * 7 * (20 + 2 * 5) = 1470 # ss 곱하기 클래수 더하고 경계박스 곱하기 5 # s * s * (c + b * 5) = 1470 
        )

def test(S=7, B=2, C=20): #테스트 함수 
    model = YOLOv1(split_size= S , num_boxes= B, num_classes=C)
    x = torch.randn((2, 3, 448, 448)) # 2 = 배치사이즈 # 3 = 채널수 # 448 = 이미지크기
    print(model(x).shape)
test()
# mode = test()

# print(test())