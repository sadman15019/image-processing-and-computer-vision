import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

img=cv2.imread("sample1.png",cv2.IMREAD_GRAYSCALE)
val,thres_img=cv2.threshold(img,100,255,cv2.THRESH_BINARY)

img_h=img.shape[0]
img_w=img.shape[1]

k_size=21
padding=k_size//2

img=cv2.copyMakeBorder(thres_img,padding,padding,padding,padding,cv2.BORDER_CONSTANT)

out=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)

struct_elem=np.ones((k_size,k_size),dtype=np.float32)

for i in range(padding,img.shape[0]-padding):
    for j in range(padding,img.shape[0]-padding):
        temp=img[i-padding:i+padding+1,j-padding:j+padding+1]
        pro=temp*struct_elem
        out[i,j]=np.min(pro)
        

cv2.imshow("out",out)

cv2.waitKey(0)