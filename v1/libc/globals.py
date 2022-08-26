# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:14:57 2022

@author: user
"""

def initialize(): 
    global modbus_ch1
    global modbus_ch2
    global modbus_ch3
    global state
    global frame
    global fsm_cnter
    global mod_en
    
    # modbus
    modbus_ch1 = 0
    modbus_ch2 = 0
    modbus_ch3 = 0
    
    # modbus
    mod_en = False
    
    # ocr
    ocr_result = ""
    # ocr_en = False
    
    fsm_cnter = 0
    
    state = "READY"
    frame = ""
