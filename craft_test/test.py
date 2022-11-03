"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from craft import CRAFT
from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v): #turns string into boolean #트룰 푸르스로 바꿔주는 함수
    return v.lower() in ("yes", "y", "true", "t", "1")

#parser 라는 객체를 만들어서 그 안에 인자들을 넣어준다.
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference') #이미지를 받아서 텍스트를 인식하는 함수
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner') 
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
 
args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder) #이미지 리스트를 받아온다.

result_folder = './result/' #결과를 저장할 폴더를 만든다.
if not os.path.isdir(result_folder): #만약 결과를 저장할 폴더가 없다면
    os.mkdir(result_folder) # 결과를 저장할 폴더가 없으면 만들어준다.

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None): #이미지를 받아서 텍스트를 인식하는 함수
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio   #이미지 비율을 구한다.

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized) #이미지를 정규화함
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w] # 4차원으로 만들어준다. 탠서로 만들어줌  
    if cuda: 
        x = x.cuda() #cuda를 사용하면 x를 cuda로 만들어준다.

    # forward pass
    with torch.no_grad(): #테스트기 때문에 역전파를 하지 않는다.
        y, feature = net(x) #

    # make score and link map # 점수와 링크 맵을 만든다.
    score_text = y[0,:,:,0].cpu().data.numpy() #텍스트를 인식하는 점수를 구한다.
    score_link = y[0,:,:,1].cpu().data.numpy() #링크를 인식하는 점수를 구한다.

    # refine link
    if refine_net is not None: #만약 리파이너 네트워크가 있다면
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy() #링크를 인식하는 점수를 구한다.

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly) #텍스트를 인식하는 점수와 링크를 인식하는 점수를 받아서 박스를 만든다.
    # 
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h) # 박스의 좌표를 조정한다
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)): 
        if polys[k] is None: polys[k] = boxes[k] #만약 폴리곤이 없다면 박스를 폴리곤으로 만들어준다. 

    t1 = time.time() - t1 

    # render results (optional)
    render_img = score_text.copy() # 텍스트를 인식하는 점수를 복사한다. 
    render_img = np.hstack((render_img, score_link)) # 텍스트를 인식하는 점수와 링크를 인식하는 점수를 합친다. regin score와 affinity score를 합친다.
    ret_score_text = imgproc.cvt2HeatmapImg(render_img) # 텍스트를 인식하는 점수를 히트맵으로 만든다. 

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize 

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list): #원본 이미지를 불러온다. 
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
         
        
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))