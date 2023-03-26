import cv2
import numpy as np
import math

img = cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)

out= np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)

mx= img.max(axis=(0,1))
c = 255/(math.log(1+mx))
for x in range(0,img.shape[0]):
    for y in range(0,img.shape[1]):
        out[x,y] = (math.exp(img[x,y]) ** (1/c)) - 1

out=cv2.normalize(out,None,0,1,cv2.NORM_MINMAX)


cv2.imshow('Input', img)
cv2.imshow('Inverse_log Output',out)       

cv2.waitKey()
cv2.destroyAllWindows()