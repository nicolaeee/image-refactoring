
import cv2
import numpy as np
path =r'C:\Users\nicuuu\PycharmProjects\pythonProject\image-refactoring\SecondLab\flori.jpeg'
image1 = cv2.imread(path)
# transformarea imaginii în grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# aplicarea diferitelor metode de prag
thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
# output images
cv2.imshow('Media adaptivă', thresh1)
cv2.imshow('Gaussian adaptiv', thresh2)
# Distribuie orice utilizare a memoriei asociate
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
