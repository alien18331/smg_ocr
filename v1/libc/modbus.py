# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:44:38 2022

@author: user
"""

import time 
# import sys
# import random
# import logging
import threading
# import modbus_tk
import modbus_tk.defines as cst
# import modbus_tk.modbus as modbus
import modbus_tk.modbus_tcp as modbus_tcp

from libc import globals

# parameter
mod_client = "127.0.0.1"
slaveID = 1

mutex = threading.Lock()
# LOGGER = modbus_tk.utils.create_logger(name="console", record_format="%(message)s")

def modbusTcp(run):
    '''
    modbus_ch1: usage hours
    modbus_ch2: Ignite
    modbus_ch3: PLASMA power
    '''
    try:
        SERVER = modbus_tcp.TcpServer(address = mod_client, port = 502)
        print("modbus runnning..")
        
        # server start
        SERVER.start()
        
        # build slave
        slv = SERVER.add_slave(int(slaveID))
        slv.add_block('A', cst.HOLDING_REGISTERS, 0, 8) # adres:0, len:8
                 
        # value
        while run.is_set():
            # val = random.randint(19000,21000)                    
            slv.set_values('A', 0, globals.modbus_ch1)
            slv.set_values('A', 1, globals.modbus_ch2)
            slv.set_values('A', 2, globals.modbus_ch3)
            time.sleep(0.5)
            
    except KeyboardInterrupt:    
        SERVER._do_exit()
        SERVER.stop()
        
    finally:
        # SERVER._do_exit()
        SERVER.stop()        
    
    return 0

def pre_proc(run):
    while run.is_set():
        if(globals.mod_en):
            mutex.acquire() # lock
            globals.mod_en = False
            mutex.release() # un-lock
            
            ocr_text = globals.ocr_result
            
            if "READY" in ocr_text:
                if("hour" in ocr_text):
                    globals.modbus_ch3 = int(ocr_text[6:10])
                    
            elif "IGNITE" in ocr_text:
                globals.modbus_ch2 = int(ocr_text[-5])
                
            elif "PLASMA" in ocr_text:
                if("k" in ocr_text):
                    globals.modbus_ch1 = int(float(ocr_text[-5:-2])*100)
    return 0

if __name__ == '__main__':
    run_event = threading.Event()
    run_event.set()
    
    modbusTcp(run_event)