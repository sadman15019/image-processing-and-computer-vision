# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:52:34 2023

@author: Asus
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt

gamma=float(input("Enter gamma value\n"))
c=int(input("Enter scaling factor\n"))

img=cv2.imread('camera.jpg',cv2.IMREAD_GRAYSCALE)
output=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
for i in range(0,img.shape[0]):
    for j in  range (0,img.shape[1]):
        output[i,j]=round(c*((img[i,j]/255)**gamma))

cv2.imshow("input",img)       
cv2.imshow("output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()