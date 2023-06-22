import matplotlib.pyplot as plt
import math
import cv2
import numpy as np


img = cv2.imread("bird.jpg",0)

def min_max_norm(inp):
    i_max= inp.max()
    i_min = inp.min()
    dif= i_max-i_min
    out=np.zeros((inp.shape),np.uint8)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            
            out[i,j]=((inp[i,j]-i_min)/dif)*255
    return np.round(out).astype(np.uint8)

x=None
y=None
point_list=[]
def onclick(event):
    global x,y
    ax=event.inaxes
    if ax is not None:
        x,y=ax.transData.inverted().transform([event.x,event.y])
        x=int(round(x))
        y=int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
       (event.button, event.x, event.y, x, y))
        point_list.append((x,y))


ft =np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

mag = np.abs(ft_shift)
ang = np.angle(ft_shift)

mag_show = np.log1p(np.abs(ft_shift)+1)
mag_scaled = min_max_norm(mag_show)


plt.title("seed input")

im=plt.imshow(mag_scaled,cmap='gray')

im.figure.canvas.mpl_connect('button_press_event',onclick)

plt.show(block=True)

btr_notch=np.ones((mag.shape),np.float32)

order=1

d0=5

m=img.shape[0]//2

n=img.shape[1]//2
tmp=[]
for k in point_list:
    x1=k[1]
    y1=k[0]
    
    if(x1<=m):
      x2=m+abs(m-x1)
    else:
      x2=m-abs(m-x1) 
    if(y1<=n):
        y2=n+abs(n-y1)  
    else:
        y2=n-abs(n-y1) 
        
        
    tmp.append((x2,y2))    
        
point_list+=tmp
for u in range(btr_notch.shape[0]):
    for v in range(btr_notch.shape[1]):
        for k in point_list:
            y1=k[0]
            x1=k[1]
            
            dk1=np.sqrt((u-x1)**2+(v-y1)**2)
            dkp=np.sqrt((u+x1)**2+(v-y1)**2)
            
            tmp1=1.0/(1.0+((d0/dk1)**(2*order)))
            tmp2=1.0/(1.0+((d0/dkp)**(2*order)))
            
            tmp1=tmp1*tmp2
            
            btr_notch[u,v]*=tmp1
            
#Btr_notch = min_max_norm(Btr_notch)

cv2.imshow("filter",btr_notch)
ang=np.angle(ft_shift)
output=mag*btr_notch
## phase add
final_result = np.multiply(output, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_norm(img_back)

## plot
cv2.imshow("input", img)


cv2.imshow("Inverse transform",img_back_scaled)       
cv2.waitKey(0)
cv2.destroyAllWindows()
            
            
        
    