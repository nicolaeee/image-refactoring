import cv2
path = r'C:\Users\ng165\PycharmProjects\pythonProject1\chess.jpg'
img = cv2.imread(path)
status = cv2.imwrite(r'C:\Users\ng165\PycharmProjects\pythonProject1\cheZZ.png', img)
print("Image written to file-system : ", status)
cv2.imshow('Lena', img)