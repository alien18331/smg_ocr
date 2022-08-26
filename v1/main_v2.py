# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:09:19 2022

@author: user
"""

# importing required libraries 
import cv2 
import time 
from threading import Thread # library for implementing multi-threaded processing 

# deskewing
import numpy as np
import math
from deskew import determine_skew
from typing import Tuple, Union

# OCR
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

# defining a helper class for implementing multi-threaded processing 
class WebcamStream :
    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.vcap.set(cv2.CAP_PROP_EXPOSURE, -4.0)
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream 
        self.stopped = True 

        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
    # method for starting the thread for grabbing next available frame in input stream 
    def start(self):
        self.stopped = False
        self.t.start() 

    # method for reading next frame 
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True 

# initializing and starting multi-threaded webcam capture input stream 
webcam_stream = WebcamStream(stream_id=0) #  stream_id = 0 is for primary camera 
webcam_stream.start()

# processing frames in input stream
num_frames_processed = 0 
start = time.time()
while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read() 
        
    # deskewing
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    skewed = rotate(grayscale, angle, (0, 0, 0))
    # skewed_rgb = cv2.cvtColor(skewed, cv2.COLOR_BGR2RGB)
    
    # region
    topx = 80 #80
    topy = 50 #40
    bottomx = 200 #125
    bottomy = 650 #310
    Cropped = skewed[topx:bottomx+1, topy:bottomy+1]
    
    # ocr
    text = pytesseract.image_to_string(Cropped, config='--psm 6')   
    print(text.strip())
    
    # adding a delay for simulating time taken for processing a frame 
    # delay = 0.03 # delay value in seconds. so, delay=1 is equivalent to 1 second 
    # time.sleep(delay) 
    num_frames_processed += 1 

    cv2.imshow('frame' , Cropped)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
end = time.time()
webcam_stream.stop() # stop the webcam stream 

# printing time elapsed and fps 
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows 
cv2.destroyAllWindows()