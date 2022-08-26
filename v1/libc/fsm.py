# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:07:33 2022

@author: user
"""
import time
import threading

from libc import globals

mutex = threading.Lock()

max_cnter = 15

def fsm(run):
    '''
    state list
    IDLE
    READY
    USAGEHR
    IGNITE
    PLASMA
    '''
    # state = "READY"
    # globals.fms_cnter = 0
    
    while run.is_set():   
        if(globals.state=="READY"):
            if(globals.fms_cnter>=max_cnter):
                mutex.acquire() # lock
                globals.state=="IDLE"
                mutex.release() # un-lock
                globals.fms_cnter = 0
            else: globals.fms_cnter = globals.fms_cnter+1
            
        elif(globals.state=="USAGEHR"):
            if(globals.fms_cnter>=max_cnter):
                mutex.acquire() # lock
                globals.state=="IDLE"
                mutex.release() # un-lock
                globals.fms_cnter = 0
            else: globals.fms_cnter = globals.fms_cnter+1
            
        elif(globals.state=="IGNITE"):
            if(globals.fms_cnter>=max_cnter):
                mutex.acquire() # lock
                globals.state=="IDLE"
                mutex.release() # un-lock
                globals.fms_cnter = 0
            else: globals.fms_cnter = globals.fms_cnter+1  
            
        elif(globals.state=="PLASMA"):
            if(globals.fms_cnter>=max_cnter):
                mutex.acquire() # lock
                globals.state=="IDLE"
                mutex.release() # un-lock
                globals.fms_cnter = 0
            else: globals.fms_cnter = globals.fms_cnter+1  
            
        else:
            mutex.acquire() # lock
            globals.state=="IDLE"
            mutex.release() # un-lock
            globals.fms_cnter = 0
        
        # print(globals.fms_cnter)
        time.sleep(1)
            
    return 0    
    
