import cv2
path =  r'C:\Users\ng165\PycharmProjects\pythonProject1\FirstLab\chess.jpg'
img = cv2.imread(path)
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
dimensions = img.shape
print('Înălțimea imaginii:', height)
print('Lățimea imaginii : ', width)
print('Numărul de canale:', channels)
print('Dimensiunile imaginii:', dimensions)
cv2.imshow("Imaginea originală", img)
#redimensionarea imaginii
imM = cv2.resize(img, (200, 400))
heightM = imM.shape[0]
widthM = imM.shape[1]
channelsM = imM.shape[2]
dimensionsM = imM.shape
print('Înălțimea imaginii redimensionate: ', heightM)
print('Lățimea imaginii redimensionate: ', widthM)
print('Numărul de canale redimensionate:', channelsM)
print('Dimensiunile imaginii redimensionate:', dimensionsM)
cv2.imshow("Imaginea redimensionata", imM)
cv2.waitKey()
cv2.destroyAllWindows()