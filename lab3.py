import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread("histogram.jpg",cv2.IMREAD_GRAYSCALE)
i = cv2.equalizeHist(img)
cv2.imshow("j",i)

out=np.zeros((img.shape[0],img.shape[1]),np.float32)
freq=np.zeros(256,np.int32)
freq_out=np.zeros(256,np.int32)


for i in range (img.shape[0]):
    for j in range(img.shape[1]):
        freq[img[i,j]]+=1


pdf=np.zeros(256,np.float32)
cdf=np.zeros(256,np.float32)
pdf_out=np.zeros(256,np.float32)
cdf_out=np.zeros(256,np.float32)

for i in range(256):
    pdf[i]=freq[i]/(img.shape[0]*img.shape[1])
    
cdf[0]=pdf[0]
for i in range(1,256):
    cdf[i]=cdf[i-1]+pdf[i]



for i in range(256):
    cdf[i]=round(cdf[i]*255.0)


    

    
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        out[i,j]=int(round(cdf[img[i,j]]))

out=out/255

cv2.imshow("input",img)
cv2.imshow("output",out)   
cv2.waitKey(0)
         

out=out*255



for i in range (out.shape[0]):
    for j in range(out.shape[1]):
        freq_out[int(out[i,j])]+=1

for i in range(256):
    pdf_out[i]=freq_out[i]/(out.shape[0]*out.shape[1])
    
cdf_out[0]=pdf_out[0]
for i in range(1,256):
    cdf_out[i]=cdf_out[i-1]+pdf_out[i]
    
for i in range(256):
    cdf_out[i]=round(cdf_out[i]*255.0)
print("asdasd")


'''
plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.title("Input Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()
plt.subplot(1, 2, 2)
plt.title("output Image Histogram")
plt.hist(out.ravel(),256,[0,255])


plt.subplot(1, 2, 1)
plt.title("Input cdf")
plt.plot(cdf)
plt.show()
plt.subplot(1, 2, 2)
plt.title("output cdf")
plt.plot(cdf_out)
plt.show()



        
    
    
#plt.figure(figsize=(10, 4))

#plt.subplot(1, 2, 1)
'''
        

        
