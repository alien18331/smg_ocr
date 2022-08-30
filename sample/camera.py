# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:29:08 2022

@author: user
"""

import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
print("opened!")
#cap.set(cv2.CAP_PROP_EXPOSURE, -4.0)

#cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE); # 命名一個視窗，可不寫
try:
    while(True):
        # 擷取影像
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break        
        # 彩色轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 顯示圖片
        cv2.imshow('live', frame)
        #cv2.imshow('live', gray)
    
        # 按下 q 鍵離開迴圈
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

except KeyboardInterrupt: 
    # 釋放該攝影機裝置
    cap.release()
    cv2.destroyAllWindows()
