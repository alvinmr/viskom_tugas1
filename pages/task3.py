# manual dilation of an image

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/coins.png', 0)
# 

plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
plt.xticks([]), plt.yticks([])
plt.show()