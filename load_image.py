import numpy as np
import cv2



img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)


#out=img.copy()
out = np.zeros((520,520), dtype=np.uint8)
#print(img.max())
#print(img.min())

#cv2.imshow('output image',out)
#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        a = img.item(i,j)
#        out.itemset((i,j),255-a)
        
#cv2.imshow('output image',out)





kernel2 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])#sharpening filter 
kernel2 = np.array([[0.1111111, 0.1111111, 0.1111111],
                    [0.1111111, 0.1111111, 0.1111111],
                    [0.1111111, 0.1111111, 0.1111111]])#box blur 

kernel=np.zeros((3,3),dtype=np.float32)  
ii=2  
for i in range(3):
    jj=2
    for  j in range(3):
        kernel[i,j]=kernel2[ii,jj]
        jj-=1
    ii-=1

x=kernel2.shape[0]//2
y=kernel2.shape[1]//2
image_bordered = cv2.copyMakeBorder(src=img, top=1, bottom=1, left=1, right=1,borderType= cv2.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT 


for i in range(1,image_bordered.shape[0]-1):
    for j in range(1,image_bordered.shape[1]-1):
        s=0
        for k in range (-x,x+1):
            for l in range (-y,y+1):
                s+=(image_bordered[i+k,j+k]*kernel[k+1,l+1])
        out[i,j]=s
        
        



cv2.imshow("image",out)
cv2.waitKey(0)
cv2.destroyAllWindows()






#cv2.normalize(src,des, 0, 255, cv2.NORM_MINMAX)
#s = np.round(s).astype(np.uint8)