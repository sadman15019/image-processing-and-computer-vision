# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:46:40 2023

@author: Asus
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt




img = cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)


#out=img.copy()

#print(img.max())
#print(img.min())

#cv2.imshow('output image',out)
#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        a = img.item(i,j)
#        out.itemset((i,j),255-a)
        
#cv2.imshow('output image',out)

kernel= np.array([[0,1,0],
                  [1,-4,1],
                  [0,1,0]])#sharpening filter 
x=kernel.shape[0]//2
y=kernel.shape[1]//2
image_bordered = cv2.copyMakeBorder(src=img, top=x, bottom=x, left=x, right=x,borderType= cv2.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT 
out = np.zeros((image_bordered.shape[0],image_bordered.shape[1]), dtype=np.float32)



for i in range(x,image_bordered.shape[0]-x):
    for j in range(y,image_bordered.shape[1]-y):
        s=0
        for a in range (-x,x+1):
            for b in range (-y,y+1):
                s+=image_bordered[i-a,j-b]*kernel[a+x,b+y]
        if(s<0):
            s=0
        out[i,j]=s
        
        

final=cv2.normalize(out,None, 0, 1, cv2.NORM_MINMAX)
cv2.imshow("image",final)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.imshow(out,'gray')
plt.title("Output for laplacian " )
plt.show()








#s = np.round(s).astype(np.uint8)