import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def ripple(img,ax,ay,tx,ty):
    out=np.zeros((img.shape),dtype=np.uint8)
    for i in range(out.shape[1]):
        for j in range(out.shape[0]):
            x=i+ax*np.sin((2*np.pi*j)/tx)
            y=j+ay*np.sin((2*np.pi*i)/ty)
            
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
     
    print(out)           
    return out
            
    
img=cv2.imread("ripple.jpg")

b,g,r=cv2.split(img)

out_b=ripple(b,10,15,50,70)

out_g=ripple(g,10,15,50,70)

out_r=ripple(r,10,15,50,70)

out=cv2.merge([out_b,out_g,out_r])

cv2.imshow("out",out)

cv2.waitKey(0)