# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:35:08 2023

@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math




sigma=float(input("Enter the value of sigma\n"))

cons=2*sigma*sigma
pi=(22/7)
k_h=int(sigma*5)
k_w=int(sigma*5)

x=k_h//2
y=k_w//2

img=cv2.imread('bilateral.png',cv2.IMREAD_GRAYSCALE)
image_bordered = cv2.copyMakeBorder(src=img, top=x, bottom=x, left=y, right=y,borderType= cv2.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT 
out = np.zeros((image_bordered.shape[0],image_bordered.shape[1]), dtype=np.float32)

if(k_h%2==0):
    k_h+=1
    k_w+=1
    
kernel=np.zeros((k_h,k_w),dtype=np.float32)
for i in range (-x,x+1):
    for j in range (-y,y+1):
        temp1=(i*i)+(j*j)
        temp1/=cons
        temp2=math.exp(-(temp1))
        temp2/=(pi*cons)
        kernel[i+x][j+y]=temp2
        
for i in range(0,k_h):
    print(kernel[i])
        

for i in  range(x,img.shape[0]-x):
    for j in  range(y,img.shape[1]-y):
        tmp2=s=0
        for a in  range(-x,x+1):
            for b in range(-y,y+1):
                r1=img[i,j]
                r2=img[i-a,j-b]
                r1=int(r1)-int(r2)
                r1=math.exp(-((r1*r1)/cons))
                r1/=(pi*cons)
                tmp1=kernel[a+x,b+y]*r1
                tmp2+=tmp1
                s+=tmp1*img[i-a,j-b]
        out[i,j]=s/tmp2
        out[i,j]/=255

cv2.imshow("input",img)   
cv2.imshow("image",out)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img,'gray')

plt.title("input for bilateral: ")

plt.show()

plt.imshow(out,'gray')

plt.title("Output for bilateral: ")

plt.show()


    
        
        