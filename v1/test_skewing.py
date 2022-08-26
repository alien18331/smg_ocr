# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:14:50 2022

@author: user
"""

import cv2
import matplotlib.pyplot as plt

import math
from typing import Tuple, Union

# import imutils
import numpy as np

from deskew import determine_skew


image_file = "./capture/2022_08_16_Catupre_1.png"


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

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image):           # 加高斯噪聲
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]   # blue
            g = image[row, col, 1]   # green
            r = image[row, col, 2]   # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    # cv2.imshow("noise image", image)

''' ================ pre-processing ================ '''
image = cv2.imread(image_file)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
gray = cv2.bilateralFilter(gray, 13, 25, 25)
    
plt.imshow(gray, plt.cm.gray)
plt.show()
# 

# contour
# edged = cv2.Canny(gray, 200, 250)    
# plt.imshow(edged, plt.cm.gray)
# plt.show()


# gaussian
# gaussian_noise(image)
# dst = cv2.GaussianBlur(image, (5,5), 0) #高斯模糊
# cv2.imshow("Gaussian_Blur2", dst)
# dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# plt.imshow(dst_rgb)
# plt.show()

# deskewing
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
angle = determine_skew(grayscale)
img_skewed = rotate(image, angle, (0, 0, 0))
img_skewed_rgb = cv2.cvtColor(img_skewed, cv2.COLOR_BGR2RGB)


plt.imshow(img_skewed_rgb)
plt.show()
