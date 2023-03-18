# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:30:37 2023

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:57:32 2023

@author: Asus
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

sigma=float(input("Enter the value of sigma\n"))

cons=2*sigma*sigma

size=round(sigma*5)
if(size%2==0):
    size+=1
    
kernel=np.zeros((size,size),dtype= np.float32)

x=size//2
y=size//2
pi=(22/7)

img=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
image_bordered = cv2.copyMakeBorder(src=img, top=x, bottom=x, left=x, right=x,borderType= cv2.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT 
out = np.zeros((image_bordered.shape[0],image_bordered.shape[1]), dtype=np.float32)
o = np.zeros((img.shape[0],img.shape[1]), dtype=np.float32)
print(pi)

for i in range (-x,x+1):
    for j in range (-y,y+1):
        temp1=(i*i)+(j*j)
        temp1/=cons
        temp2=math.exp(-(temp1))
        temp2/=(pi*cons)
        kernel[i+x][j+y]=temp2
    

        
for i in range(0,size):
    print(kernel[i])

ksum=kernel.sum()
    
for i in range (x,img.shape[0]-x):
    for j in range (y,img.shape[1]-y):
        s=0
        for a in range(-x,x+1):
            for b in range(-y,y+1):
                s+= kernel[a+x,b+y]*img[i-a,j-b]
            s/=ksum
        out[i,j]=round(s)

plt.imshow(out,'gray')
plt.title("Output for gaussian blurr "+str(sigma) )
plt.show()

o=cv2.normalize(out,None, 0, 1, cv2.NORM_MINMAX)
#o=out[x-1:-x-4,y-1:-y-4]              
cv2.imshow("image",o)
cv2.waitKey(0)
cv2.destroyAllWindows()

                
    
    

        