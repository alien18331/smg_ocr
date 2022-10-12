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

import TIS

# deskewing
import numpy as np
import math
# from deskew import determine_skew # for detect angle
from typing import Tuple, Union

# OCR
import pytesseract
from libc import azure_ocr
# import pytessy

# modbus
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp

# debug
image_debug = False
# image_file = "./MKS.png"
image_file = "./img/2022_09_01_Catupre_8.png"

# parameters
config = configparser.ConfigParser()
config.read('/home/smg/smg/ocr/v5/config.ini')

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


list_KEY = ['GNITE','GHITE','ON','hour']

err_char = False
result = 0
key_type = 0
lastVal_ON = 0
lastVal_hour = 0
lastVal_IGNITE = 0
list_last = []

# list_READY = ['READY', 'EADY']
# list_IGNITE = ['IGNITE', 'GNITE']
# list_PLASMA = ['PLASMA', 'PLASHA']

# if(windows_base):
#     print("windows_base: {}".format(windows_base))
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

def merge_rgb(R,G,B):
    img = cv2.merge([B,G,R])
    return img

def img_preprocessing(image, angle):
    global region_topx
    global region_topy
    global region_bottomx
    global region_bottomy    
    
    #print("region_topx:{}".format(region_topx))
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if(image_debug):
        # print("test")
        
        skewed = rotate(grayscale, 3.5, (0, 0, 0))
        
        # region
        region_topx = 200
        region_topy = 0
        region_bottomx = 320
        region_bottomy = 650
        Cropped = skewed[region_topx:region_bottomx+1, region_topy:region_bottomy+1]
        
        # print("grayscale shape:{}".format(grayscale.shape))
        
        # blur = Cropped
        blur = cv2.fastNlMeansDenoising(Cropped, h=7,templateWindowSize=7,searchWindowSize=21);
        
        result = blur
        
    else:
        
        # Cropped
        region_topx = 170
        region_topy = 70
        region_bottomx = 300
        region_bottomy = 550
        Cropped = frame[region_topx:region_bottomx, region_topy:region_bottomy]
        # show_img(Cropped)
        
        # skew
        skewed = rotate(Cropped, 3.3, (0, 0, 0))
        # show_img(skewed)
        
        # Cropped
        region_topx = 45
        region_topy = 30
        region_bottomx = 115
        region_bottomy = 445
        Cropped = skewed[region_topx:region_bottomx, region_topy:region_bottomy]
        # show_img(Cropped)
        
        # G channel 
        (B, G, R) = cv2.split(Cropped)    
        zeros = np.ones(Cropped.shape[:2],dtype=np.uint8)
        G_ch_img = merge_rgb(R=zeros, G=G, B=zeros)
        # show_img(G_ch_img)
        
        # gray
        grayscale = cv2.cvtColor(G_ch_img, cv2.COLOR_BGR2GRAY)
        
        # blur-median
        blur_median = cv2.medianBlur(grayscale,3)
        # show_img(blur_median)
        
        # enhance black
        enhanced = np.where((blur_median>50) & (blur_median<135), blur_median-40, blur_median)
        # show_img(enhanced)
        
        
        result = enhanced
        
    return result

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def modbus_preprocessing(ocr_text):
    global scale
    global lastVal_ON
    global lastVal_hour
    global lastVal_IGNITE
    global list_last
    global key_type
    global err_char
    
    global result
    scale = 1
    st_idx = 0
    
    # case sensitive
    for key in list_KEY: #choosen key
        if(key in ocr_text): #check key
            if(key=="ON"):
                st_idx = ocr_text.find(" ")+1
                scale = 100
                key_type = 1
                if(len(ocr_text.strip())>st_idx): # "PLASMA ON 3. 7kw"
                    tmp_val = ""
                    for i in range(len(ocr_text)-st_idx):
                        tmp_char = ocr_text[st_idx+i]
                        if(tmp_char==" " and i>3):
                            break
                        elif(tmp_char.isnumeric() or tmp_char=="."): #fetch if char is valid
                            tmp_val = tmp_val+tmp_char
                    if(tmp_val[-1:]=="."): #remove last char if non-numeric
                        tmp_val = tmp_val[:-1]
                    if(isfloat(tmp_val)): #chech float or not
                        list_last.append(float(tmp_val)*scale)
                        print(list_last)
                        
            elif(key=="hour"):
                st_idx = ocr_text.find(" ")+1
                scale = 1
                key_type = 2
                if(len(ocr_text.strip())>st_idx): # "READY 14996. hour"
                    tmp_val = ""
                    for i in range(len(ocr_text)-st_idx):
                        tmp_char = ocr_text[st_idx+i]
                        if(tmp_char==" " and i>3):
                            break
                        elif(tmp_char.isnumeric() or tmp_char=="."): #fetch if char is valid
                            tmp_val = tmp_val+tmp_char
                    
                    if(tmp_val[-1:]=="."): #remove last char if non-numeric
                        tmp_val = tmp_val[:-1]
                    if(isfloat(tmp_val)): #chech float or not
                        list_last.append(float(tmp_val)*scale)
                        print(list_last)
                        
            elif(key=="GNITE" or key=='GHITE'):
                st_idx = ocr_text.find(" ")+1
                scale = 1
                key_type = 3
                if(len(ocr_text.strip())>st_idx): # "READY 14996. hour"
                    tmp_val = ""
                    for i in range(len(ocr_text)-st_idx):
                        tmp_char = ocr_text[st_idx+i]
                        if(tmp_char==" " and i>3):
                            break
                        elif(tmp_char.isnumeric() or tmp_char=="."): #fetch if char is valid
                            tmp_val = tmp_val+tmp_char
                        else:
                            if(err_char):
                                break
                            err_char = True
                    
                    if(tmp_val[-1:]=="."): #remove last char if non-numeric
                        tmp_val = tmp_val[:-1]
                    if(isfloat(tmp_val)): #chech float or not
                        list_last.append(float(tmp_val)*scale)
                        print(list_last)
                        
        elif(key==list_KEY[-1]): #end
            #print("result")
            lastVal = 0
            if(key_type==1):
                lastVal = lastVal_ON
                
            elif(key_type==2):
                lastVal = lastVal_hour
                
            elif(key_type==3):
                lastVal = lastVal_IGNITE
            
            
            if((lastVal*scale) in list_last): # choose lastVal as output
                result = lastVal
            else: # update lastVal and choose mode value
                for i in list_last:
                    counts = np.bincount(list_last)
                    result = (np.argmax(counts)/scale)
                    lastVal = (result/scale)
            
            # reset
            #key_type = 0
            list_last = []
        
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
       
# initializing and starting multi-threaded webcam capture input stream 
# webcam_stream = WebcamStream(stream_id=0) #  stream_id = 0 is for primary camera 
# webcam_stream.start()

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

Tis = TIS.TIS()

# The following line opens and configures the video capture device.
Tis.openDevice("16024087", 640, 480, "30/1",TIS.SinkFormats.BGRA, False)

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
    Tis.Set_Property("BalanceWhiteAuto", "Continuous")
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

    Tis.Set_Property("ExposureTime", 24000)

except Exception as error:
    print(error)
    quit()    

error = 0
print('Press Esc to stop')
lastkey = 0
# cv2.namedWindow('Window',cv2.WINDOW_NORMAL)

modbus = ModbusTCP(client=mod_client, port=mod_port, slaveID=slaveID, size=size)
modbus.start()

if(debug):
    # processing frames in input stream
    num_frames_processed = 0 
    start = time.time()

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
            frame = CD.image
            
    # while True :
    #     if webcam_stream.stopped is True :
    #         break
    #     else :
    #         frame = webcam_stream.read() 
            
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
            tmpVal = modbus_preprocessing(ocr_text)
            print("ocr_value: {}".format(tmpVal))
            
            # modbus
            if(debug):
                ocr_val = random.uniform(3.5,4.5)*100
            # print(ocr_text)
                
            #modbus.update(ocr_val)
            
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
    # webcam_stream.stop() 
    Tis.Stop_pipeline()
    modbus.stop()
    
    if(debug):
        # printing time elapsed and fps 
        elapsed = end-start
        fps = num_frames_processed/elapsed 
        print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))
        
        if(wind_show):
            # closing all windows 
            cv2.destroyAllWindows()
        #cv2.destroyAllWindows()
