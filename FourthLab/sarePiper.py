import cv2
import numpy as np
from skimage.util import random_noise
import matplotlib.pyplot as plt
from scipy import signal
from skimage import filters
from PIL import Image, ImageFilter
path = r'C:\Users\nicuuu\PycharmProjects\pythonProject\image-refactoring\FourthLab\butterfly.jpg'
img = cv2.imread(path,0)
noise_img = random_noise(img, mode='s&p', amount=0.3)
medie7 = cv2.blur(img,(11,11))
gauss = np.array([[0,1,1,2,2,2,1,1,0],
                  [1,2,4,5,5,5,4,2,1],
                  [1,4,5,3,0,3,5,4,1],
                  [2,5,3,-12,-24,-12,3,5,2],
                  [2,5,0,-24,-40,-24,0,5,2],
                  [2,5,3,-12,-24,-12,3,5,2],
                  [1,4,5,3,0,3,5,4,1],
                  [1,2,4,5,5,5,4,2,1],
                  [0,1,1,2,2,2,1,1,0]])
I = signal.convolve2d(img, gauss, boundary='symm', mode='same')
edge_scharr = filters.scharr(img)
plt.subplot(231),plt.imshow(img),plt.title('Imagine Originala')
plt.subplot(232),plt.imshow(noise_img),plt.title('Zgomot S&P')
plt.subplot(233),plt.imshow(medie7),plt.title('Medie 11x11')
plt.subplot(234),plt.imshow(I),plt.title('Filtrul Log')
plt.subplot(235),plt.imshow(edge_scharr),plt.title('Filtrul Scharr')
plt.show()