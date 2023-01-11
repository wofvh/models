# 텍스트 파일에서 원하는 글자 제거하기 (remove text from a text file) 


import os
import os
import shutil
import json
from PIL import Image 
import numpy as np


with open("C:\gt.txt", "r",encoding='cp949') as f:
    lines = f.readlines()

list = []

for line in lines:
    line = line.replace("test/", "")
    print(line)
    list.append(line)
    
with open("C:\gt3.txt", "w",encoding='cp949') as f:
    for line in list:
        if line.strip("\n") != "test":     # <= 이 문자열만 골라서 삭제
            f.write(line)
