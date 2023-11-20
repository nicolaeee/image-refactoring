import cv2
import numpy as np

path = r'C:\Users\nicuuu\PycharmProjects\pythonProject\image-refactoring\FourthLab\cheZZ4.png'
img1 = cv2.imread(path)

if img1 is None:
    print(f"Error: Unable to load image at path: {path}")
else:
    print(f"Image loaded successfully. Shape: {img1.shape}")
    gauss = np.random.normal(0, 1, img1.size)
    gauss = gauss.reshape(img1.shape[0], img1.shape[1], img1.shape[2]).astype('uint8')
    zgomot = img1 + img1 * gauss
    # display the image
    cv2.imshow('a', zgomot)
    cv2.waitKey(0)
