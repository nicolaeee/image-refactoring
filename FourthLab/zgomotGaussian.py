import cv2
import numpy as np
img = cv2.imread('bmw4.jpg')
# generam zgomont Gaussian in mod aleatoriu
mean = 0
stddev = 180
noise = np.zeros(img.shape, np.uint8)
cv2.randn(noise, mean, stddev)
# adaugam zgomot
noisy_img = cv2.add(img, noise)
cv2.imshow('zgomot',noisy_img)
cv2.waitKey(0)