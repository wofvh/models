import os
import shutil
import json
from PIL import Image 
import numpy as np

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def read_all_file(path):
     output = os.listdir(path)
     file_list = []

     for i in output:
         if os.path.isdir(path+"\\"+i): #isdir은 폴더인지 확인
             file_list.extend(read_all_file(path+"\\"+i)) #extend는 리스트에 리스트를 추가
         elif os.path.isfile(path+"\\"+i): #isfile은 파일인지 확인
             file_list.append(path+"\\"+i)

     return file_list
           
w= 720 #이미지 가로크기
h= 540 #이미지 세로크기
date = '/IMG_OCR_53_4PO_04726/' #내 데이터가 날짜로 되어있음
json_path = 'C://coco128//test//labels//train2017//' + date # 데이터 있는 기존파일(json)
file_list = read_all_file(json_path) #json 파일 읽기


for i in file_list:   #파일 루프
    with open(i, 'rt', encoding='UTF8') as f:
        ex1 = json.load(f)
        
        for i in range(len(ex1['annotations'])):
            x = ''
            output = ''
            print()
            
            for j in range (4):
                x = x + str(ex1['annotations'][i]['points'][j])
                point_o = ex1['annotations'][i]['points']
        
            box = (point_o[0][0], point_o[2][0], point_o[0][1], point_o[2][1])
        
            change_box = convert((w,h),box)
           
            output = str(label2num(ex1['annotations'][i]["label"])) + ' '  + str(change_box[0]) + ' ' + str(change_box[1])+ ' ' + str(change_box[2])+ ' ' + str(change_box[3]) +'\n'
            
            #print(output)
            print(output)
            
            with open ( f.name[:-8] + 'txt'     , 'a+') as file:
                file.write(output)
                file.close()
