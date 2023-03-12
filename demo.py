# Importing the OpenCV, Numpy and Mat libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image from the disk using cv2.imread() function
# Showing the original image using matplotlib library function plt.imshow()
img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
print(img)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Kernel for box blur filter
# It is a unity matrix which is divided by 9
box_blur_ker = np.array([[0.1111111, 0.1111111, 0.1111111],
					[0.1111111, 0.1111111, 0.1111111],
					[0.1111111, 0.1111111, 0.1111111]])

# Applying Box Blur effect
# Using the cv2.filter2D() function
# src is the source of image(here, img)
# ddepth is destination depth. -1 will mean output image will have same depth as input image
# kernel is used for specifying the kernel operation (here, box_blur_ker)
Box_blur = cv2.filter2D(src=img, ddepth=-1, kernel=box_blur_ker)

# Showing the box blur image using matplotlib library function plt.imshow()
print(Box_blur)
cv2.imshow("image",Box_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.show()
