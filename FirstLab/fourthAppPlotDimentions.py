import cv2
img = cv2.imread('chess.jpg')
from matplotlib import pyplot as plt
bicubic_img = cv2.resize(img, None, fx = 2, fy = 2 , interpolation = cv2.INTER_CUBIC)
near_img = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_NEAREST)
bilinear_img = cv2.resize(img,None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
cv2.imshow("Imaginea originală", bicubic_img)
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Imagine originala')
plt.subplot(222), plt.imshow(bicubic_img , 'gray'), plt.title('bicubic')
plt.subplot(223), plt.imshow(bilinear_img, 'gray'), plt.title('cel mai apropiat vecin')
plt.subplot(224), plt.imshow(bilinear_img, 'gray'), plt.title('bilinara')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()