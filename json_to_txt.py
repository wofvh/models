import re
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Json 파일들 상위 디렉토리
input_dir = 'C:\\bae' 

def convert(size, coord_list):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (coord_list[0] + coord_list[1]) / 2.0
    y = (coord_list[2] + coord_list[3]) / 2.0
    w = coord_list[1] - coord_list[0]
    h = coord_list[3] - coord_list[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

file_list = os.listdir(input_dir)

json_name_list = [file for file in file_list if file.endswith('.json')]

for i in range(len(json_name_list)):
    json_name_list[i] = input_dir + '/' + json_name_list[i]

count = 0

for i in json_name_list:
    output_file_name = i.split('.')[0] + '.txt'
    with open(i, 'r', encoding='UTF-8') as file:
        data = file.read().replace('\n', '')
    count_data = data.count('\"data\"')
    height = re.findall('\d+', data.split('height')[1].split(',')[0]) #findall은 정규식에 맞는 모든 문자열을 리스트로 반환
    width = re.findall('\d+', data.split('width')[1].split(',')[0])
    size = [width[0], height[0]] #size = [width, height]
    size =  list(map(int, size)) #map은 리스트의 요소를 지정된 함수로 처리해주는 함수
    data = data.split("bbox")[1] #bbox를 기준으로 나눔 #bbox가 없는 경우가 있음 
    data_list = data.split("}")
    result = []
    for j in range(count_data):
        temp_data = data_list[j]
        data_name = temp_data.split('\"data\":')[1].split(',')[0]
        x_list = temp_data.split(',      \"y\": ')[0].split('"x": ')[1].replace('[', '').replace(']', '').replace(' ', '').split(',')
        y_list = temp_data.split(',      \"y\": ')[1].replace('[', '').replace(']', '').replace(' ', '').split(',')
        coord_list = [x_list[0], x_list[-1], y_list[0], y_list[-1]]
        coord_list =  list(map(int, coord_list))
        yolo_style = list(convert(size, coord_list))
        yolo_style.insert(0, data_name)
        result.append(' '.join(map(str, yolo_style)))

    for n in range(len(result)):
        result[n] = result[n].strip()

    #       #txt 파일 새로만든 폴더로 저장
    # with open('C:\\newone/' + output_file_name.split('/')[-1], 'w+') as lf: #새로운 폴더에 txt 파일로 저장
    #     lf.write('\n'.join(result))

    with open(output_file_name, 'w+') as lf:
        lf.write('\n'.join(result))

    with open(output_file_name, 'r') as lf:
        readList = lf.readlines()

    le = LabelEncoder()
    output_file_name = le.fit_transform([readList] + ['\n']) 