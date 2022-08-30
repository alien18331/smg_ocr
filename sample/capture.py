# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:29:08 2022

@author: user
"""

import sys
import cv2
import datetime

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

#cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE); # 命名一個視窗，可不寫
idx = 1
today = datetime.date.today()

# cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4.0) # diable auto exposure

try:
    while(True):
        # 擷取影像
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        # 彩色轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Cropped
        # topx = 200 #80
        # topy = 120 #40
        # bottomx = 380 #125
        # bottomy = 540 #310
        # Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    
        # 顯示圖片
        cv2.imshow('live', gray)
        #cv2.imshow('live', gray)        
        
        # 按下 q 鍵離開迴圈
        if cv2.waitKey(1) == ord('c'):            
            filename = "./{}_Catupre_{}.png".format(today.strftime('%Y_%m_%d'), idx)
            # saving image in local storage
            cv2.imwrite(filename, frame)  
            idx = idx + 1
        
        elif cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

except KeyboardInterrupt: 
    # 釋放該攝影機裝置
    cap.release()
    cv2.destroyAllWindows()
