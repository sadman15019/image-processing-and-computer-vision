# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:19:36 2023

@author: ASUS
"""

import numpy as np
import cv2



img = cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)

#out=img.copy()


kernel = np.array([[1, 2, 3,4,5],
                    [6, 7, 8,9,10],
                    [11, 12, 1,-12,-11],
                    [-10, -9, -8,-7,-6],
                    [-5, -4, -3,-2,-1]])

m=1 #center
n=1 #center

img_bordered=cv2.copyMakeBorder(img,top=3,bottom=1,left=3,right=1,borderType=cv2.BORDER_CONSTANT)

out = np.zeros((img_bordered.shape), dtype=np.float32)

for i in range(3,img_bordered.shape[0]-1):
    for j in range(3,img_bordered.shape[1]-1):
        su=0
        for s in range(-1,4):
            for t in range(-1,4):
                su+=img_bordered[i-s,j-t]*kernel[s+1,t+1]
          
                
        out[i,j]=su
    
print(out)
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
print(out)
out= np.round(out).astype(np.uint8)
print(out)
cv2.imshow("output",out)
cv2.waitKey(0)                
                