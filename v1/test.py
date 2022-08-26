# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:46:09 2022

@author: user
"""

import cv2
import imutils
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

from PIL import Image
from libc import azure_ocr

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

lib_azure_ocr = 0
fn_equalization = 1

''' ================ image ================ '''
# image source
# img_path = "./MKS_Test.png"
image_file = "./MKS_Test.png"
# img_path = "./car.png"

''' ================ pre-processing ================ '''
img = cv2.imread(image_file)
# img = cv2.resize(img, (600,400))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray = cv2.bilateralFilter(gray, 13, 15, 15)


# edged = cv2.Canny(gray, 30, 150)
# plt.imshow(edged, plt.cm.gray)
# plt.show()

# contour
# contours = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# contours = sorted(contours,key=cv2.contourArea, reverse = True)[:20]
# screenCnt = None
# for c in contours:
#     peri = cv2.arcLength(c, True) # object 周長
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#     if len(approx) == 6:
#         screenCnt = approx
#         break
# print("screenCnt: {0}".format(screenCnt))

mask = np.zeros(gray.shape,np.uint8)
# new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
# new_image = cv2.bitwise_and(img,img,mask=mask)

# plt.imshow(new_image, plt.cm.gray)

# region    
(x, y) = np.where(mask == 255)
# (topx, topy) = (np.min(x), np.min(y))
# (bottomx, bottomy) = (np.max(x), np.max(y))
topx = 80
topy = 40
bottomx = 125
bottomy = 310
Cropped = gray[topx:bottomx+1, topy:bottomy+1]
plt.imshow(Cropped, plt.cm.gray)


# equalization
eq = cv2.equalizeHist(Cropped) # BGR
eq_RGB = cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)
plt.imshow(eq_RGB, plt.cm.gray)

if(lib_azure_ocr): 
    preproced_img = Image.open(image_file)
else:
    preproced_img = eq_RGB

''' ================ OCR ================ '''
if(lib_azure_ocr):    
    result_dict = azure_ocr.azure_ocr(preproced_img)
    
else: # local OCR
    text = pytesseract.image_to_string(preproced_img, config='--psm 6')    
    # print("\nprogramming_fever's License Plate Recognition")
    # print("Detected license plate Number is: {0}".format(text.strip()))
    
    result_dict = {}
    result_dict[0] = text.strip()
    
print(result_dict)  


''' ================ modbus ================ '''