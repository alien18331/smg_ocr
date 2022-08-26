# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:10:18 2022

@author: user
"""

import csv
import cv2
from pytesseract import pytesseract as pt

file = './mks.png'

img = cv2.imread(file)
h, w, _ = img.shape

boxes = pt.image_to_boxes(img)

for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow(file, img)
cv2.waitKey(0)