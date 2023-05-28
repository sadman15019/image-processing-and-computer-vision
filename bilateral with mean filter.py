
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

sigma=float(input("Enter the value of gaussian sigma\n"))

cons=2*sigma*sigma

size=5
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

out/=255


o=cv2.normalize(out,None, 0, 1, cv2.NORM_MINMAX)
#o=out[x-1:-x-4,y-1:-y-4]              
#cv2.imshow("input",img)
#cv2.imshow("image",o)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#size=int(input("Enter kernel size\n"))

pi=22/7
img=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
kernel2=np.array([[1/25,1/25,1/25,1/25,1/25],
          [1/25,1/25,1/25,1/25,1/25],
          [1/25,1/25,1/25,1/25,1/25],
          [1/25,1/25,1/25,1/25,1/25],
          [1/25,1/25,1/25,1/25,1/25]])

x=5//2
y=5//2
image_bordered = cv2.copyMakeBorder(src=img, top=x, bottom=x, left=x, right=x,borderType= cv2.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT 
out2 = np.zeros((image_bordered.shape[0],image_bordered.shape[0]), dtype=np.float32)
#o = np.zeros((img.shape[0],img.shape[0]), dtype=np.uint8)

#print(img.shape)
#print(image_bordered.shape)
'''
for i in range (x,img.shape[0]-x):
    for j in range (y,img.shape[1]-y):
        s=0
        for a in range(-x,x+1):
            for b in range(-y,y+1):
                s+= kernel2[a+x,b+y]*img[i-a,j-b]
            s/=25
        out2[i,j]=s
        out2[i,j]=s/255
        '''
s=int(input("enter the value for bilateral sigma\n"))
cons=2*s*s      
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
                tmp1=kernel2[a+x,b+y]*r1
                tmp2+=tmp1
                s+=tmp1*img[i-a,j-b]
        out2[i,j]=s/tmp2
       # print(out2)
        out2[i,j]/=255

#o=out[1:-1,1:-1]  
cv2.imshow("input",img)
cv2.imshow("gausian output",out)               
cv2.imshow("bilateral output",out2)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(out2,'gray')

plt.title("Output for bilateral: ")

plt.show()

                
                
    
    

        