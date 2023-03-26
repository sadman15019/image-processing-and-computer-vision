import cv2 
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('lena_mean.jpg',cv2.IMREAD_GRAYSCALE)

size=int(input("Enter kernel size\n"))




x=size//2
y=size//2
image_bordered = cv2.copyMakeBorder(src=img, top=x, bottom=x, left=x, right=x,borderType= cv2.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT 
out = np.zeros((image_bordered.shape[0],image_bordered.shape[0]), dtype=np.uint8)
o = np.zeros((img.shape[0],img.shape[0]), dtype=np.uint8)

#print(img.shape)
#print(image_bordered.shape)

for i in range(x,image_bordered.shape[0]-x):
    for j in range(y,image_bordered.shape[1]-y):
        s=0
        for k in range(i-x,i+x+1):
            for l in range(j-x,j+x+1):
                s+=image_bordered[k,l]
                out[i,j]=round(s/(size*size))
o=out[1:-1,1:-1]  
cv2.imshow("input",img)               
cv2.imshow("image",o)
cv2.waitKey(0)
cv2.destroyAllWindows()

                