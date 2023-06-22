import cv2 
import matplotlib.pyplot as plt
import numpy as np

def eq(img):
    #plt.hist(img.ravel(),256,(0,256))
    
    out=np.zeros((img.shape),np.uint8)
    
    freq=np.zeros(256,np.float32)
    pdf=np.zeros(256,np.float32)
    cdf=np.zeros(256,np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            freq[img[i,j]]+=1
            
    pdf=freq/(img.shape[0]*img.shape[1])
    cdf[0]=pdf[0]
    for i in range(1,256):
        cdf[i]=cdf[i-1]+pdf[i]
    
    for i in range(256):
        cdf[i]=round(cdf[i]*255)
        
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j]=cdf[img[i,j]]
            
    return out

img=cv2.imread("bd.jpeg")

# plt.subplot(3, 2, 1)
# plt.title("input channel histogram")
# histr, _ = np.histogram(img[:,:,2],256,[0,256])
# plt.plot(histr,color = 'r')
# histg, _ = np.histogram(img[:,:,1],256,[0,256])
# plt.plot(histg,color = 'g')
# histb, _ = np.histogram(img[:,:,0],256,[0,256])
# plt.plot(histb,color = 'b')

b,g,r=cv2.split(img)

out_b=eq(b)
out_g=eq(g)
out_r=eq(r)

out=cv2.merge([out_b,out_g,out_r])

# plt.subplot(3, 2, 1)
# plt.title("output channel histogram")
# histr, _ = np.histogram(out[:,:,2],256,[0,256])
# plt.plot(histr,color = 'r')
# histg, _ = np.histogram(out[:,:,1],256,[0,256])
# plt.plot(histg,color = 'g')
# histb, _ = np.histogram(out[:,:,0],256,[0,256])
# plt.plot(histb,color = 'b')

cv2.imshow("out",out)
cv2.waitKey(0)


