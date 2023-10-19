import cv2
import numpy as np
from matplotlib import pyplot as plt
path =r'C:\Users\nicuuu\PycharmProjects\pythonProject\image-refactoring\SecondLab\bmw.jpg'
img = cv2.imread(path)
(thresh1, I1) = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
(thresh2,I2) = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
(thresh3,I3) = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
(thresh4,I4) = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
(thresh5,I5) = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, I1, I2, I3, I4, I5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    plt.xlim([0, 256])
    plt.show()
    cv2.waitKey(1000)
