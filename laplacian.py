# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:19:36 2023

@author: ASUS
"""

import numpy as np
import cv2

def minmax(img):
    x=np.min(img)
    y=np.max(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]=((img[i,j]-x)/(y-x))*255
            
  
    return img

img = cv2.imread('moon.png',cv2.IMREAD_GRAYSCALE)


kernel= np.array([[0,-1,0],
                  [-1,4,-1],
                  [0,-1,0]])#sharpening filter 

x=kernel.shape[0]//2
y=kernel.shape[1]//2
#bordered_img = cv2.copyMakeBorder(img,1,3,1,3,cv2.BORDER_CONSTANT,0)
borderd_image=cv2.copyMakeBorder(src=img,top=x,bottom=x,left=y,right=y,borderType=cv2.BORDER_CONSTANT)

#cv2.imshow('bordered image',bordered_img)
out=np.zeros((borderd_image.shape),dtype=np.float32)



for i in range(x,borderd_image.shape[0]-x):
    for j in range(y,borderd_image.shape[1]-y):
        s=0
        for a in range(-x,x+1):
            for b in range(-y,y+1):
                s+=borderd_image[i-a,j-b]*kernel[a+x,b+y]
                
        out[i,j]=s
  
out=minmax(out)       
out=borderd_image+out   
    
final=minmax(out) 

final/=255

cv2.imshow("input",img)
cv2.imshow("output",final)
cv2.waitKey(0)


#cv2.normalize(src,des, 0, 255, cv2.NORM_MINMAX)
#s = np.round(s).astype(np.uint8)