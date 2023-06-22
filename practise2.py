import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
import math
#matplotlib.use('TkAgg')

def angular(img):
    out=np.zeros((img.shape),dtype=np.uint8)
    xc=img.shape[1]//2
    yc=img.shape[0]//2
    
    #rmax=min(xc,yc)
    a=0.3
    tau=50
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
           dx= i-xc
           dy= j=yc
           r=np.sqrt(dx*dx+dy*dy)
           print(r)
           beta=math.atan2(dy,dx)+a*np.sin(( 2*np.pi*r )/tau)
           print(beta)
           x=xc+r*np.cos(beta)
           y=yc+r*np.sin(beta)
        
           jj=int(np.floor(x))
           kk=int(np.floor(y))
           a=x-jj
           b=y-kk
           if(jj<img.shape[1]-1 and kk<img.shape[0]-1):
               arr=[[img[kk][jj],img[kk][jj+1]],
                    [img[kk+1][jj],img[kk+1][jj+1]]]
               temp2=[[1-b,b]]
               temp1=[[1-a],
                      [a]]
               ans=np.matmul(arr,temp1)
               ans=np.matmul(temp2,ans)
               ans=ans[0][0]
           else:
               ans=0
               
           out[j][i]=ans
    
        
    return out
           
           
           
    

img = cv2.imread("twirl.jpg")


b,g,r=cv2.split(img)

b_out=angular(b)

g_out=angular(g)

r_out=angular(r)

output1 = cv2.merge([b_out,g_out,r_out])

cv2.imshow("output angular",output1)
cv2.waitKey(0)
cv2.destroyAllWindows()

