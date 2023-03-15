# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:15:02 2023

@author: Asus
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt

Xmax=0
Xmin=255

img=cv2.imread('lena2.jpg',cv2.IMREAD_GRAYSCALE)
output=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
for i in range(0,img.shape[0]):
    for j in  range (0,img.shape[1]):
        Xmax=max(Xmax,img[i,j])
        Xmin=min(Xmin,img[i,j])
        
for i in range(0,img.shape[0]):
    for j in  range (0,img.shape[1]):
        output[i,j]=((img[i,j]-Xmin)/(Xmax-Xmin))*255
       
cv2.imshow("output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()