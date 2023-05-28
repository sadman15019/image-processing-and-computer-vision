# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
import math



def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input

img_input = cv2.imread('homo.jpg', 0)

img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac=np.abs(ft_shift)

m=img.shape[0]//2
n=img.shape[1]//2

magnitude_spectrum = 1 * np.log(np.abs(ft_shift)+1)

magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)


ang = np.angle(ft_shift)



filt = np.zeros((img.shape[0],img.shape[1]), np.float32)

gh = 1.5
gl = 0.6
c = 0.05
d0 = 50

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        u = (i-img.shape[0]/2)
        v = (j-img.shape[1]/2)
        r = math.exp(-c*((u*u+v*v)/(d0**2)))
        r = (gh-gl)*(1-r)+gl
        filt[i][j] = r
        

        
temp=filt*np.abs(ft_shift)
temp2= 1 * np.log(np.abs(temp)+1)
temp2 = min_max_normalize(temp)
cv2.imshow("filter",filt)
cv2.imshow("Magnitude Spectrum after filtering",temp)
## phase add
final_result = np.multiply(temp, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)


## plot

cv2.imshow("inputtt", img_input)
cv2.imshow("Magnitude Spectrummmm",magnitude_spectrum_scaled)

cv2.imshow("Phaseee",ang)
cv2.imshow("Inverse transform",img_back_scaled)




cv2.waitKey(0)
cv2.destroyAllWindows() 