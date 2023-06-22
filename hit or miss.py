import cv2
import matplotlib.pyplot as plt
import numpy as np

def hitormiss(b1,w,img):
    tmp1=cv2.erode(img,b1,iterations=1)
    b2=w-b1
    com=cv2.bitwise_not(img)
    tmp2=cv2.erode(com,b2,iterations=1)
    out=cv2.bitwise_and(tmp1,tmp2)
    return out

img=cv2.imread("hit or miss.jpg",0)

r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)#_INV)

cv2.imshow("input",img)



x1=np.array(([0,0,0],
             [1,1,0],
             [1,0,0]),dtype=np.uint8)
            
x2=np.array(([0,1,1,],
             [0,0,1],
             [0,0,1]),dtype=np.uint8)

x3 = np.array(([1,1,1],
               [0,1,0],
               [0,1,0]),np.uint8)

w=np.array(([1,1,1],
            [1,1,1],
            [1,1,1]),dtype=np.uint8)

rate=50
x1=cv2.resize(x1,None,fx=rate,fy=rate,interpolation=cv2.INTER_NEAREST)

x2=cv2.resize(x2,None,fx=rate,fy=rate,interpolation=cv2.INTER_NEAREST)

x3=cv2.resize(x3,None,fx=rate,fy=rate,interpolation=cv2.INTER_NEAREST)

w=cv2.resize(w,None,fx=rate,fy=rate,interpolation=cv2.INTER_NEAREST)

out=hitormiss(x3*255,w*255,img)

#out2=hitormiss(x2*255,w*255,img)

#out3=hitormiss(x3*255,w*255,img)



cv2.imshow("output1",out)
#cv2.imshow("output2",out1)
#cv2.imshow("output3",out1)
cv2.waitKey(0)
cv2.destroyAllWindows()


            