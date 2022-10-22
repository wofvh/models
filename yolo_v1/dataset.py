"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,  #csv 파일, 이미지 폴더, 라벨 폴더, S 분할 수, B 박스 수, C 클래스 수
    ):
        self.annotations = pd.read_csv(csv_file) #csv 파일 읽기 
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations) #길이를 반환하는 함수

    def __getitem__(self, index): #인덱스를 받아서 해당 인덱스의 이미지와 라벨을 반환하는 함수
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) #라벨 파일 경로 #Join은 경로를 합쳐주는 함수
        boxes = []
        with open(label_path) as f:  #라벨 파일을 읽어서 박스를 만듬 열린 레이블 파일을 f에 저장
            for label in f.readlines():  #라벨 파일을 한줄씩 읽어서
                class_label, x, y, width, height = [  #클레스 라벨, x좌표, y좌표, 너비, 높이를 저장 
                    float(x) if float(x) != int(float(x)) else int(x)  # x 는 소수점이될수도 있어서  #x가 정수가 아니면 float로 저장
                    for x in label.replace("\n", "").split() #라벨 파일에서 \n을 제거하고 공백을 기준으로 나눔
                ]

                boxes.append([class_label, x, y, width, height]) #클래스 라벨, x좌표, y좌표, 너비, 높이를 boxes에 저장

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) #이미지 파일 경로 iloc(로케이션)
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self. transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes) #이미지와 박스를 변환

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) #라벨 행렬을 왜 0으로 초기화 하지?
        for box in boxes:
            class_label, x, y, width, height = box.tolist() #박스를 리스트로 변환 #tolist는 텐서를 리스트로 변환
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
 
            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j 
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:  # i ,j 는 행과 열 객채가 
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix