import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


img=cv2.imread("eye.png",cv2.IMREAD_GRAYSCALE)
freq=np.zeros(256,np.int32)
freq=cv2.calcHist([img],[0],None,[256],[0,256])


out=np.zeros((img.shape[0],img.shape[1]),np.float32)

pdf=np.zeros(256,np.float32)
cdf=np.zeros(256,np.float32)
pdf_erlang=np.zeros(256,np.float32)
cdf_erlang=np.zeros(256,np.float32)
tmp=np.zeros(256,np.int32)

for i in range(256):
    pdf[i]=freq[i]/(img.shape[0]*img.shape[1])
    

    
cdf[0]=pdf[0]
for i in range(1,256):
    cdf[i]=cdf[i-1]+pdf[i]
    



for i in range(256):
    cdf[i]=round(cdf[i]*255.0)




'''
cv2.imshow("input",img)
cv2.imshow("output",out)   

'''


k=int(input(("Enter shape parameter\n")))
miu=float(input(("Enter scale parameter\n")))

lamda=1.0/miu



for i in range(0,256):
    pdf_erlang[i]=(math.pow(lamda,k)*math.pow(i,k-1))
    pdf_erlang[i]*=np.exp(-lamda*i)
    pdf_erlang[i]/=math.factorial(k-1)
    



#plt.hist(pdf_erlang.ravel(),256,[0,255])
    
cdf_erlang[0]=pdf[0]
for i in range(1,256):
    cdf_erlang[i]=cdf_erlang[i-1]+pdf_erlang[i]
    

for i in range(256):
     cdf_erlang[i]=round(cdf_erlang[i]*255.0)   

print(cdf_erlang)
for i in range(256):
    x=cdf[i]
    mn=10000000000
    for j in range(256):
        if(abs(x-cdf_erlang[j])<mn):
            index=j
            mn=abs(x-cdf_erlang[j])
    tmp[i]=cdf_erlang[index]
print(tmp)
    
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        out[i,j]=int(tmp[img[i,j]])
        

histr = cv2.calcHist([out],[0],None,[256],[0,256])
plt.plot(histr)


out=out/255
cv2.imshow("input",img)
cv2.imshow("output",out)   
cv2.waitKey(0)

plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.title("Input Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()
plt.subplot(1, 2, 2)
plt.title("output Image Histogram")
plt.plot(histr)  

#plt.hist(cdf_erlang.ravel(),256,[0,255])


    
    



