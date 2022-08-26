# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:06:25 2022

@author: user
"""

import sys
import cv2
import time
import datetime
import logging
import threading

import modbus_tk
import pytesseract

from PIL import Image

from libc import azure_ocr
from libc import globals
from libc import pre_proc
from libc import modbus

# debug
import random
import matplotlib.pyplot as plt

# global 
globals.initialize() 

# logger
logger = modbus_tk.utils.create_logger(name="console", record_format="%(message)s")

# parameters
debug = 1
lib_azure_ocr = 0
fn_equalization = 1

# thread event
run_event = threading.Event()
run_event.set()

# thread modbus-modbusTcp
# th_modbusTcp = threading.Thread(target = modbus.modbusTcp, args=(run_event,))
# th_modbusTcp.start()

# thread modbus-pre_proc
th_modbus_preproc = threading.Thread(target = modbus.pre_proc, args=(run_event,))
th_modbus_preproc.start()

''' camera '''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open camera")
    sys.exit()

else:
    time.sleep(2)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4.0) # diable auto exposure
    
    try:
        while run_event.is_set():        
            ''' camera thread '''
            ret, frame = cap.read() # BGR chammel
            if not ret:
                logger.error("Can't receive frame (stream end?). Exiting ...")
                run_event.clear()
                break        
            
            ''' image pre-processing '''
            if(lib_azure_ocr): 
                # preproced_img = Image.open(image_file)
                tmp_result = pre_proc.img_processing(frame)
                preproced_img = Image.fromarray(cv2.cvtColor(tmp_result, cv2.COLOR_BGR2RGB))  
            else:
                preproced_img = pre_proc.img_processing(frame)
            
            #debug
            if(debug):
                cv2.imshow('live', preproced_img)
            cv2.imshow('live', preproced_img)
            
            ''' OCR '''
            # if(lib_azure_ocr):    
            #     ocr_text = azure_ocr.azure_ocr(preproced_img)  
            # else: # smg OCR
            #     ocr_text = pytesseract.image_to_string(preproced_img, config='--psm 6') 
            # print(ocr_text.strip())
             
            #debug
            if(debug):
                ocr_text = "PLASMA ON {:.2f}k".format(random.uniform(3.5,4.5))
            
            globals.ocr_result = ocr_text.strip()
            
            # ''' modbus '''
            # globals.mod_en = True
            
            #debug
            if(debug):
                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
    
    except KeyboardInterrupt:    
        if(debug):
            cv2.destroyAllWindows()
        run_event.clear()
        cap.release()
        
    finally:
        cap.release()
    
    print("Done.")