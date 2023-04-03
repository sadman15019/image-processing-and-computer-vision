# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:04:42 2023

@author: Asus
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

img=cv2.imread("histogram.jpg",cv2.IMREAD_GRAYSCALE)
out=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
plt.hist(img.ravel(),256,(0,256))

freq=np.zeros(256,np.int32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        freq[img[i,j]]+=1

        
plt.hist(freq,256,(0,256))

pdf=np.zeros(256,np.float32)
cdf=np.zeros(256,np.float32)
for i in range(256):
    pdf[i]=freq[i]/(img.shape[0]*img.shape[1])
    

cdf[0]=pdf[0]
    
for i in range(1,256):
    cdf[i]=cdf[i-1]+pdf[i]
    
for i in range(256):
    cdf[i] = round(cdf[i]*255.0)
    

    
  
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        out[i,j]=int(round(cdf[img[i,j]]))

plt.hist(out.ravel(),256,(0,256))
out/=255       
cv2.imshow("input",img)  
cv2.imshow("output",out)  
cv2.waitKey(0)
cv2.destroyAllWindows()
        



    
    
