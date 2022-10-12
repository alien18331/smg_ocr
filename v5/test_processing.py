#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:27:48 2022

@author: smg
Note: Save an image on trigger
"""

# Add path to python-common/TIS.py to the import path
import cv2
import numpy as np
import os
import sys
import time
sys.path.append("../python-common")

import TIS

import math
from typing import Tuple, Union
import matplotlib.pyplot as plt
import pytesseract


# This sample shows, how to get an image in a callback and use trigger or software trigger
# needed packages:
# pyhton-opencv
# pyhton-gst-1.0
# tiscamera

cam_show = True


class CustomData:
    ''' Example class for user data passed to the on new image callback function
    '''
    def __init__(self, newImageReceived, image):
        self.newImageReceived = newImageReceived
        self.image = image
        self.busy = False

CD = CustomData(False, None)

def on_new_image(tis, userdata):
    '''
    Callback function, which will be called by the TIS class
    :param tis: the camera TIS class, that calls this callback
    :param userdata: This is a class with user data, filled by this call.
    :return:
    '''
    # Avoid being called, while the callback is busy
    if userdata.busy is True:
        return

    userdata.busy = True
    userdata.newImageReceived = True
    userdata.image = tis.Get_image()
    userdata.busy = False
    

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

Tis = TIS.TIS()

# The following line opens and configures the video capture device.
# office: 16024087
# fab: 49914015

Tis.openDevice("49914015", 640, 480, "30/1",TIS.SinkFormats.BGRA, False) # video live

# The next line is for selecting a device, video format and frame rate.
# if not Tis.selectDevice():
#     quit(0)

#Tis.List_Properties()
Tis.Set_Image_Callback(on_new_image, CD)

Tis.Set_Property("TriggerMode", "On")

Tis.Start_pipeline()

# Remove comment below in oder to get a propety list.
# Tis.List_Properties()

# In case a color camera is used, the white balance automatic must be
# disabled, because this does not work good in trigger mode
try:
    Tis.Set_Property("BalanceWhiteAuto", "True")
    # Tis.Set_Property("BalanceWhiteAuto", "Off")
    # Tis.Set_Property("BalanceWhiteRed", 1.2)
    # Tis.Set_Property("BalanceWhiteGreen", 1.0)
    # Tis.Set_Property("BalanceWhiteBlue", 1.4)
    
except Exception as error:
    print(error)

try:
    # Query the gain auto and current value :
    print("GainAuto : %s " % Tis.Get_Property("GainAuto"))
    print("Gain : %d" % Tis.Get_Property("Gain"))

    # Check, whether gain auto is enabled. If so, disable it.
    if Tis.Get_Property("GainAuto"):
        Tis.Set_Property("GainAuto", "Off")
        print("Gain Auto now : %s " % Tis.Get_Property("GainAuto"))

    Tis.Set_Property("Gain", 0)

    # Now do the same with exposure. Disable automatic if it was enabled
    # then set an exposure time.
    if Tis.Get_Property("ExposureAuto") :
        Tis.Set_Property("ExposureAuto", "Off")
        print("Exposure Auto now : %s " % Tis.Get_Property("ExposureAuto"))

    Tis.Set_Property("ExposureTime", 64000)

except Exception as error:
    print(error)
    quit()    

error = 0
print('Press Esc to stop')
lastkey = 0
if(cam_show): cv2.namedWindow('Window') # cv2.WINDOW_NORMAL

try:
    while lastkey != 27 and error < 5:
        # time.sleep(1)
        Tis.execute_command("TriggerSoftware") # Send a software trigger

        # Wait for a new image. Use 10 tries.
        tries = 10
        while CD.newImageReceived is False and tries > 0:
            time.sleep(0.1)
            tries -= 1

        # Check, whether there is a new image and handle it.
        if CD.newImageReceived is True:
            CD.newImageReceived = False
            
            # print("image fetching")
            frame = CD.image[:,:,:3]
            
            ''' processing '''
            
            # Cropped
            region_topx = 170
            region_topy = 75
            region_bottomx = 300
            region_bottomy = 560
            Cropped = frame[region_topx:region_bottomx, region_topy:region_bottomy]
            # if(img_plot): show_img(Cropped)
            
            # skew
            skewed = rotate(Cropped, 3.3, (0, 0, 0))
            # if(img_plot): show_img(skewed)
            
            # Cropped
            region_topx = 45
            region_topy = 30
            region_bottomx = 115
            region_bottomy = 455
            Cropped = skewed[region_topx:region_bottomx, region_topy:region_bottomy]
            # if(img_plot): show_img(Cropped)
            
            # G channel 
            (B, G, R) = cv2.split(Cropped)    
            zeros = np.ones(Cropped.shape[:2],dtype=np.uint8)
            G_ch_img = merge_rgb(R=zeros, G=G, B=zeros)
            # if(img_plot): show_img(G_ch_img)
            
            # gray
            grayscale = cv2.cvtColor(G_ch_img, cv2.COLOR_BGR2GRAY)
            
            # blur-median
            blur_median = cv2.medianBlur(grayscale,3)
            # if(img_plot): show_img(blur_median)
            rgb = cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB)
            #plt.imsave(rgb, "fab")
            #cv2.imwrite('fab.png', blur_median)  
            
            
            # enhance black
            enhanced = np.where((blur_median>30) & (blur_median<60), blur_median-25, blur_median)
            # if(img_plot): show_img(enhanced)
            
            # INV
            _,th_inv = cv2.threshold(enhanced,65,255,cv2.THRESH_BINARY)
            # if(img_plot): show_img(th_inv)
            
            result = enhanced
            
            # warpAffine
            shift = 4
            p1 = np.float32([[shift,0],[result.shape[1],0],[0,result.shape[0]]])
            p2 = np.float32([[0,0],[result.shape[1]-shift,0],[0,result.shape[0]]])
            M = cv2.getAffineTransform(p1,p2)
            output = cv2.warpAffine(result, M, (result.shape[1],result.shape[0]))
            # show_img(output)
            
            # OCR
            ocr_type = "--psm {}".format(7)
            ocr_text = pytesseract.image_to_string(output, config=ocr_type)   
            
            print(ocr_text.strip())
            
            
            ''' processing End '''
            
            if(cam_show): cv2.imshow('Window', output)
        else:
            print("No image received")

        if(cam_show): lastkey = cv2.waitKey(10)

except KeyboardInterrupt:
    if(cam_show): cv2.destroyWindow('Window')
    print("terminated..!")

# Stop the pipeline and clean ip
Tis.Stop_pipeline()
if(cam_show): cv2.destroyAllWindows()
print('Program ends')
