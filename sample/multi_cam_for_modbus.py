# importing required libraries 

import sys
import cv2 
import time 
import random
import threading
from threading import Thread # library for implementing multi-threaded processing 

# modbus
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp

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

mutex = threading.Lock()

# initializing and starting multi-threaded webcam capture input stream 
webcam_stream_0 = WebcamStream(stream_id=0) #  stream_id = 0 is for primary camera 
webcam_stream_0.start()

webcam_stream_1 = WebcamStream(stream_id=2) #  stream_id = 0 is for primary camera 
webcam_stream_1.start()

modbus = ModbusTCP(client='127.0.0.1', port=502, slaveID=1, size=8)
modbus.start()

# processing frames in input stream
num_frames_processed = 0 
start = time.time()
while True :
    if webcam_stream_0.stopped or webcam_stream_0.stopped is True :
        break
    else :
        frame0 = webcam_stream_0.read() 
        frame1 = webcam_stream_1.read() 
        
    # process
    
    val = random.uniform(3.5,4.5)*100
    modbus.update(val)
    
    
    num_frames_processed += 1 

    cv2.imshow('frame' , frame0)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
end = time.time()
webcam_stream_0.stop() # stop the webcam stream 
webcam_stream_1.stop() 

# printing time elapsed and fps 
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows 
cv2.destroyAllWindows()
