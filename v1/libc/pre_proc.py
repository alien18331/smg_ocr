# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:17:41 2022

@author: user
"""
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

# deskewing
import math
from deskew import determine_skew
from typing import Tuple, Union

camera = 1 # 0:image, 1:camera

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def img_processing(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 150)
    
    # plt.imshow(edged, plt.cm.gray)
    # plt.show()
    
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    # plt.imshow(rgb_frame)
    # plt.show()
    
    
    # deskewing
    angle = determine_skew(gray)
    img_skewed = rotate(image, angle, (0, 0, 0))
    img_skewed_rgb = cv2.cvtColor(img_skewed, cv2.COLOR_BGR2RGB)
    
    # contour
    # contours = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # contours = sorted(contours,key=cv2.contourArea, reverse = True)[:20]
    # screenCnt = None
    # for c in contours:
    #     peri = cv2.arcLength(c, True) # object 周長
    #     approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    #     if len(approx) == 4:
    #         screenCnt = approx
    #         break
    # print("screenCnt: {0}".format(screenCnt))
    
    # mask = np.zeros(gray.shape,np.uint8)
    # new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    # new_image = cv2.bitwise_and(img,img,mask=mask)
    
    # plt.imshow(new_image, plt.cm.gray)
    
    # region    
    # (x, y) = np.where(mask == 255)
    # (topx, topy) = (np.min(x), np.min(y))
    # (bottomx, bottomy) = (np.max(x), np.max(y))
    topx = 250 #80
    topy = 120 #40
    bottomx = 430 #125
    bottomy = 540 #310
    Cropped = img_skewed_rgb[topx:bottomx+1, topy:bottomy+1]
    # plt.imshow(Cropped, plt.cm.gray)
    
    
    # equalization
    eq = cv2.equalizeHist(Cropped) # BGR
    # eq_RGB = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)
    # plt.imshow(eq_RGB, plt.cm.gray)
        
    return eq

if __name__ == '__main__':
    # load image
    if(camera):
        cap = cv2.VideoCapture(0)        
        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()
        
        cap.set(cv2.CAP_PROP_EXPOSURE, -4.0) # diable auto exposure
        
        while(True):
            # 擷取影像
            ret, frame = cap.read()
                
    else:
        image_bgr = cv2.imread("../MKS_Test.png")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 
        
        plt.imshow(image)
        plt.show()
    
    proced_img = img_processing(image)
    plt.imshow(proced_img)
    plt.show()
    