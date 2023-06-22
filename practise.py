import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("eye.png",cv2.IMREAD_GRAYSCALE)


plt.hist(img.ravel(),256,[0,255])

plt.show()

cv2.waitKey(0)