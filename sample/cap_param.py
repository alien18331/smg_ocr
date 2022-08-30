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
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, 100)
cap.set(14, 480) # Gain
cap.set(44, 0) # auto WB

#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
#cap.set(cv2.CAP_PROP_EXPOSURE, float(0.2))
#cap.set(14, 480)

#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#cap.set(cv2.CAP_PROP_EXPOSURE, 200)

#cap.set(cv2.CAP_PROP_GAIN, 0.0)

for i in range(47):
	print("NO.={}, parameter={}".format(i, cap.get(i)))
	
while True:
	ret, img = cap.read()
	cv2.imshow("input", img)
	
	if cv2.waitKey(1) == ord('q'):		
		break

cap.release()
cv2.destroyAllWindows()
