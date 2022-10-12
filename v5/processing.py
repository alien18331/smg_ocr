#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 08:34:12 2022

@author: smg
"""

import cv2
import math
import numpy as np
from typing import Tuple, Union
import matplotlib.pyplot as plt

import pytesseract

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

def show_his(img):
    color = ('b','g','r')
    plt.style.use('dark_background')
    plt.figure(figsize=(10,5))
    for idx, color in enumerate(color):
        his = cv2.calcHist([img],[idx],None,[256],[0,256])
        plt.plot(his, color = color)
        plt.xlim([0,256])
        
    plt.show()

def merge_rgb(R,G,B):
    img = cv2.merge([B,G,R])
    return img

def show_img(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

def split_rgb(img):
    (B, G, R) = cv2.split(img)
    
    zeros = np.ones(img.shape[:2],dtype=np.uint8)
    
    print("R Channel:")
    merge_img = merge_rgb(R=R, G=zeros, B=zeros)
    gray = cv2.cvtColor(merge_img, cv2.COLOR_BGR2GRAY)
    show_img(gray)
    print("G Channel:")
    merge_img = merge_rgb(R=zeros, G=G, B=zeros)
    gray = cv2.cvtColor(merge_img, cv2.COLOR_BGR2GRAY)
    show_img(gray)
    print("B Channel:")
    merge_img = merge_rgb(R=zeros, G=zeros, B=B)
    gray = cv2.cvtColor(merge_img, cv2.COLOR_BGR2GRAY)
    show_img(gray)

image_file = "./fab.png"
img_plot = True


frame = cv2.imread(image_file) 
if(img_plot): show_img(frame)

# split_rgb(frame_rgb)

# Cropped
# region_topx = 170
# region_topy = 70
# region_bottomx = 300
# region_bottomy = 550
# Cropped = frame[region_topx:region_bottomx, region_topy:region_bottomy]
# if(img_plot): show_img(Cropped)

# skew
# skewed = rotate(Cropped, 3.3, (0, 0, 0))
# if(img_plot): show_img(skewed)

# Cropped
# region_topx = 45
# region_topy = 30
# region_bottomx = 115
# region_bottomy = 445
# Cropped = skewed[region_topx:region_bottomx, region_topy:region_bottomy]
# if(img_plot): show_img(Cropped)

# G channel 
# (B, G, R) = cv2.split(Cropped)    
# zeros = np.ones(Cropped.shape[:2],dtype=np.uint8)
# G_ch_img = merge_rgb(R=zeros, G=G, B=zeros)
# if(img_plot): show_img(G_ch_img)

# gray
# grayscale = cv2.cvtColor(G_ch_img, cv2.COLOR_BGR2GRAY)
grayscale = frame

# blur-median
# blur_median = cv2.medianBlur(grayscale,3)
blur_median = grayscale
if(img_plot): show_img(blur_median)

# enhance black
enhanced = np.where((blur_median>30) & (blur_median<60), blur_median-25, blur_median)
if(img_plot): show_img(enhanced)

# equalization
# eq = cv2.equalizeHist(blur_median)
# show_img(eq)

# binary
# _,th_bin = cv2.threshold(croped_rgb,205,255,cv2.THRESH_BINARY)
# th_bin_rgb = cv2.cvtColor(th_bin, cv2.COLOR_BGR2RGB)
# plt.imshow(th_bin_rgb)
# plt.show()

# blur_median
# blur_median = cv2.medianBlur(croped_rgb,3)
# blur_median_rgb = cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB)
# plt.imshow(blur_median_rgb)
# plt.show()

# blur_gaussian
# blur_gau = cv2.GaussianBlur(th_bin,(11,11),0)
# blur_gau_rgb = cv2.cvtColor(blur_gau, cv2.COLOR_BGR2RGB)
# plt.imshow(blur_gau_rgb)
# plt.show()

# INV
_,th_inv = cv2.threshold(enhanced,60,255,cv2.THRESH_BINARY)
if(img_plot): show_img(th_inv)



# erode
# kernel = np.ones((3,3), np.uint8)
# erosion = cv2.erode(th_inv, kernel, iterations = 1)
# if(img_plot): show_img(erosion)

# dilate
# kernel = np.ones((3,3), np.uint8)
# dilation = cv2.dilate(erosion, kernel, iterations = 1)
# if(img_plot): show_img(dilation)


# Trunc
# _,th_trunc = cv2.threshold(croped_rgb,215,255,cv2.THRESH_TRUNC)
# th_trunc_gray = cv2.cvtColor(th_trunc, cv2.COLOR_BGR2GRAY)
# th_trunc_rgb = cv2.cvtColor(th_trunc, cv2.COLOR_BGR2RGB)
# plt.imshow(th_trunc_rgb)
# plt.show()

# threshold
# th_gau = cv2.adaptiveThreshold(Cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6)
# th_gau_rgb = cv2.cvtColor(th_gau, cv2.COLOR_BGR2RGB)
# plt.imshow(th_gau_rgb)
# plt.show()

# Canny
# canny = cv2.Canny(Cropped,50,200)
# canny_rgb = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
# plt.imshow(canny_rgb)
# plt.show()

# equalization
# eq = cv2.equalizeHist(th_trunc_gray)
# plt.imshow(eq, 'gray')
# plt.show()

result = th_inv

# blur
# blur = cv2.fastNlMeansDenoising(eq, h=7,templateWindowSize=7,searchWindowSize=21);
# blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
# plt.imshow(blur_rgb)
# plt.show()

# warpAffine
shift = 4
p1 = np.float32([[shift,0],[result.shape[1],0],[0,result.shape[0]]])
p2 = np.float32([[0,0],[result.shape[1]-shift,0],[0,result.shape[0]]])
M = cv2.getAffineTransform(p1,p2)
output = cv2.warpAffine(result, M, (result.shape[1],result.shape[0]))
show_img(output)

# OCR
ocr_type = "--psm {}".format(8)
ocr_text = pytesseract.image_to_string(output, config=ocr_type)   

print(ocr_text.strip())
