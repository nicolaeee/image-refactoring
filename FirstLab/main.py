import cv2
path =  r'C:\Users\ng165\PycharmProjects\pythonProject1\FirstLab\chess.jpg'
img = cv2.imread(path)
cv2.imshow('Image', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()