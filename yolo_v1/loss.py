 #yolo  loss function

from msilib.schema import Class
import torch
from torch import nn
from utils import intersection_over_union # iou를 구하는 함수를 가져온다.

class YoloLoss(nn.Module): 
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__() 
        self.mse = nn.MSELoss(reduction="sum") #x,y, w,h loss 임 축소 # sum 전부 더한다. #검은 배경이 있으니 mse로 계산
        self.S = S # grid 한개의 크기 = (448/64) S = 7. 30
        self.B = B # 각각의 grid cell에서 나오는 output의 matrix의 행이 B인듯
        self.C = C # 그림과 label 인식. class
        self.lambda_noobj = 0.5 # no obj 가 나왔을때 loss 가 더 크게 (나올 확률이 적어서) 더 크게
        self.lambda_coord = 5 # loss = (pred_box(x,y,w,h) - true_box(x,y,w,h))**2 # 1꼭짓점에 대한 높이와 너비

    def forward(self, predictions, target): 
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5) # prediction 는 0~ cell*cell 보 일꺼니까(cell*cell*30) 표준폼으로 펼친다.(boksu) 

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[...,21:25]) #p1/b1 target #0~19는 class , 21~25는 첫 번쨰 상자의 대한 4개의 경계 상자 값 #20은 class 점수
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[...,21:25]) #p2/b2 #26~30 는 2번째 상자의, target은 하나니까 b1 이다.
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # (2,7,7) # unsqueeze: 차원 늘려주고. / cat: 2개의 iou 상자를 이어 붙인다. #차원0에다가 붙인다.
        iou_maxes, bestbox = torch.max(ious, dim=0) # (7,7) 2개의 iou상자들 중에 max (conf확률을 넣겠다.)
        exists_box = target[..., 20].unsqueeze(3)# #.zeros(predictions[...,20]) # G=(x,y,w,h,class-conf) (7,7,30) 마지막 dims을 늘린것. # target[..., 20]= (7,7) iou가 maximun인 상자가 교차된거니까 bestbox과 같다.

        # ==================== #
        #   FOR Coordinate LOSS    #
        # ==================== #

        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30] # 교차되는 bestbox 인자에 맞출 것
                + (1 - bestbox) * predictions[..., 21:25] # 교차되지 않는 bestbox 인자에 맞춰 (해당 0 Bbox 혹은 1 Bbox) (선택/둘다 선택 X)
            )
        )
        box_targets = exists_box * target[..., 21:25] # (7,7,4)= G(x,y,w,h)= (x,y,w,h,class-conf)

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt( # sqrt: 제곱 root
            torch.abs(box_predictions[..., 2:4] + 1e-6) # 연산돌릴 때 0 경계조건 안띄게(비정상적인 숫자가 안나오게) 1e-6 아슬아슬하게
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4]) # G(x,y,w,h) = #타겟의 교정값(경계상 양수?)#
        
        # (N, S, S, B, 4) > # (N, S*S*B, 4)    
        box_loss = self.lambda_coord * self.mse(     # x,y,w,h coord 값 loss MSE
            torch.flatten(box_predictions, start_dim=0, end_dim=2), # 교정된 픽셀에 대한 플랫 가중치를 찾는다.
            torch.flatten(box_targets, start_dim=0, end_dim=2), 
        )
        
        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        #Confidence Loss

        # pred_box는 가장 높은 ioU를 가지는 BBox의 confidence score 값. # bestbox == max IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21] #슬라이스 차원에 맞게 나눈다 차원을 유지하기 위해 !
        ) # bestbox 에서 상자 1개만 1이고 다른녀석들은 0인 텐서 / 1-bestbox는 피처 맵에서 그 상자가 선택되는 두번째 상자이다.
        
        #(N, S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box), # flatten: (S*S*B) pred_box=상자의 크기 [24] [40] [53] 이렇게 있음 3개의 상자 # 상자 사이즈 = conf loss
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        # (N, S, S, 1) > (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1), # [6.09]에 있음
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1), # 선택이 됬는지 안됬는지 0으로 잡고 ==차원을 줄이기 위함
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1), # [28.21]에 있음    
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1) # 엑스티츠가 나가야하는데 predictions 에는 없는경우 #target은 0으로 잡고 
        ) 

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),   # 마지막 index 바로 전까지 20까지이므로(21 로 선택하면 안된는 class 인스턴스) -2 .
            torch.flatten(exists_box * target[..., :20], end_dim=-2,), 
        )

        #lambda   
        loss = ( 
            self.lambda_coord * box_loss           # (S*S*B,4) 0.5 x (x .x) = ([12]: [29]): [57]: [55]: [11]: [53]: [48]: [46]
            + object_loss  # third row in paper [29.76] # 그리고 2번째 오브젝트 지역에서는 감지확률 아주 작은것으로 계산을 돌린다. !
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss

modeld = YoloLoss()

print(modeld)