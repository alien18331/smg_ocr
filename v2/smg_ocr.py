#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:09:19 2022

@author: user
"""

# importing required libraries 
import sys
import cv2 
import time 
import random
import configparser
import matplotlib.pyplot as plt
from threading import Thread # library for implementing multi-threaded processing 

# deskewing
import numpy as np
import math
# from deskew import determine_skew # for detect angle
from typing import Tuple, Union

# OCR
import pytesseract
from libc import azure_ocr

# modbus
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp

# debug
image_debug = False
image_file = "./MKS.png"

# parameters
config = configparser.ConfigParser()
config.read('/home/smg/smg/ocr/v2/config.ini')

windows_base    = config['general'].getboolean('windows_base')
debug           = config['general'].getboolean('debug')

# ocr
azure_ocr_en    = config['ocr'].getboolean('azure_ocr_en')
win_psm         = config['ocr'].getint('win_psm')

# camera
cameraID = config['camera'].getint('cameraID')
wind_show = config['camera'].getboolean('windows_show')

# image pre-processing
angle           = config['img_preprocessing'].getfloat('angle')
region_topx     = config['img_preprocessing'].getint('region_topx')
region_topy     = config['img_preprocessing'].getint('region_topy')
region_bottomx  = config['img_preprocessing'].getint('region_bottomx')
region_bottomy  = config['img_preprocessing'].getint('region_bottomy')

# modbus
mod_client  = config['modbus']['mod_client']
mod_port    = config['modbus'].getint('mod_port')
slaveID     = config['modbus'].getint('slaveID')
size        = config['modbus'].getint('size')
scale       = config['modbus'].getint('scale')

if(windows_base):
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

def img_preprocessing(image, angle):
    global region_topx
    global region_topy
    global region_bottomx
    global region_bottomy    
    
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if(image_debug):
        # region
        region_topx = 80
        region_topy = 40
        region_bottomx = 125
        region_bottomy = 310
        Cropped = grayscale[region_topx:region_bottomx+1, region_topy:region_bottomy+1]
        
        result = Cropped
        
    else:
        # deskewing        
        # angle = determine_skew(grayscale)
        skewed = rotate(grayscale, angle, (0, 0, 0))
        # skewed_rgb = cv2.cvtColor(skewed, cv2.COLOR_BGR2RGB)
        
        # region
        # region_topx = 80 #80
        # region_topy = 50 #40
        # region_bottomx = 220 #125
        # region_bottomy = 650 #310
        Cropped = skewed[region_topx:region_bottomx+1, region_topy:region_bottomy+1]
        
        # equalization
        # img_eq = cv2.equalizeHist(Cropped) # BGR
        
        result = Cropped
        
    return result

def modbus_preprocessing(ocr_text):
    global scale
    
    if "READY" in ocr_text:
        if("hour" in ocr_text):
            result = int(ocr_text[6:10])
            
    elif "IGNITE" in ocr_text:
        result = int(ocr_text[-5])
        
    elif "PLASMA" in ocr_text:
        idx_st  = ocr_text.find('ON')+2
        idx_end = ocr_text.find('k')
        tmp_val = ocr_text[idx_st:idx_end].strip()
        
        val = ""
        for char in tmp_val:
            if(char==" "):
                continue
            val = val + char
        val = float(val)  
        result = int(val*scale)
            
    return result

# define ModbusTCP with multi-threaded processing
class ModbusTCP:
    def __init__(self, client='127.0.0.1', port=502, slaveID=1, size=8):        
        self.client = client
        self.slaveID = slaveID
        self.port = port
        self.size = size
        self.val = 0
        
        # create modbus server
        self.SERVER = modbus_tcp.TcpServer(address = '127.0.0.1', port = int(self.port))
        
        # server start
        self.SERVER.start()
        
        # print("slaveID: {}, size: {}".format(self.slaveID, self.size))
        # build slave
        self.slave = self.SERVER.add_slave(int(self.slaveID))
        self.slave.add_block('A', cst.HOLDING_REGISTERS, 0, self.size) # adres:0, len:8
        
        self.stopped = True 
        
        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.holdingRegister, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
    # start
    def start(self):
        self.stopped = False
        self.t.start() 
        
    # update 
    def holdingRegister(self):        
        while True:        
            if(self.stopped):
                break
            else:
                self.slave.set_values('A', 0, int(self.val))
            time.sleep(0.5)
        
        self.SERVER.stop()
    
    def update(self, value):        
        self.val = value        
    
    def stop(self):
        self.stopped = True 
       
        
# defining a helper class for implementing multi-threaded processing 
class WebcamStream :
    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            sys.exit(0)
        self.vcap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.vcap.set(cv2.CAP_PROP_EXPOSURE, 100)
        self.vcap.set(14, 480) # Gain
        self.vcap.set(44, 0) # auto WB
        
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            sys.exit(0)

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

modbus = ModbusTCP(client=mod_client, port=mod_port, slaveID=slaveID, size=size)
modbus.start()

if(debug):
    # processing frames in input stream
    num_frames_processed = 0 
    start = time.time()

try:
    while True :
        if webcam_stream.stopped is True :
            break
        else :
            frame = webcam_stream.read() 
            
        if(image_debug):
            frame = cv2.imread(image_file)        
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            # gray = cv2.bilateralFilter(gray, 13, 15, 15)
            
        # image pre-processing
        proced_img = img_preprocessing(frame, angle)
        
        # plt.imshow(proced_img)
            
        # OCR
        if(azure_ocr_en):
            ocr_text = azure_ocr.azure_ocr(proced_img)          
        else: # local ocr
            ocr_type = "--psm {}".format(win_psm)
            ocr_text = pytesseract.image_to_string(proced_img, config=ocr_type)   
        
        print(ocr_text.strip())
        
        # modbus pre-processing
        #ocr_val = modbus_preprocessing(ocr_text)
        #print("ocr_value: {}".format(ocr_val))
        
        # modbus
        if(debug):
            ocr_val = random.uniform(3.5,4.5)*100
        # print(ocr_text)
        modbus.update(ocr_val)
        
        if(debug):
        # adding a delay for simulating time taken for processing a frame 
        # delay = 0.03 # delay value in seconds. so, delay=1 is equivalent to 1 second 
        # time.sleep(delay) 
            num_frames_processed += 1 
            
            if(wind_show):
                cv2.imshow('frame' , proced_img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            
except KeyboardInterrupt:    
    print("User Terminated..!")
    # webcam_stream.stop() 
    # modbus.stop()
    
finally:
    if(debug):
        end = time.time()
    
    # terminate process
    print("Stop the threading..!")
    webcam_stream.stop() 
    modbus.stop()
    
    if(debug):
        # printing time elapsed and fps 
        elapsed = end-start
        fps = num_frames_processed/elapsed 
        print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))
        
        if(wind_show):
            # closing all windows 
            cv2.destroyAllWindows()
