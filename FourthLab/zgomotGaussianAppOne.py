import cv2
import numpy as np
from scipy import ndimage

# Incarcă imaginea
img = cv2.imread('bmw4.jpg')

# Verifică dacă imaginea a fost încărcată cu succes
if img is None:
    print("Eroare: Nu s-a putut încărca imaginea.")
else:
    # Generează zgomot Gaussian
    medie = 0
    deviatie_standard = 180
    zgomot = np.zeros(img.shape, np.uint8)
    cv2.randn(zgomot, medie, deviatie_standard)

    # Adaugă zgomot Gaussian la imagine
    imagine_zgomotata = cv2.add(img, zgomot)

    # Afișează imaginea originală și cea cu zgomot alături
    cv2.imshow('Imagine Originală', img)
    cv2.imshow('Imagine cu Zgomot', imagine_zgomotata)

    # Aplică filtre
    # 2.c.1: Aplică un filtru trece-jos (estompare)
    imagine_blurata = cv2.blur(imagine_zgomotata, (5, 5))

    # 2.c.2: Aplică un filtru statistic (median)
    imagine_median = cv2.medianBlur(imagine_zgomotata, 5)

    # 2.c.3: Aplică un filtru derivativ de ordinul I
    imagine_derivativa = cv2.Sobel(imagine_zgomotata, cv2.CV_64F, 1, 0, ksize=3)

    # 2.c.4: Aplică un filtru derivativ de ordinul II (Laplacian)
    imagine_laplaciana = cv2.Laplacian(imagine_zgomotata, cv2.CV_64F)

    # Afișează imaginile filtrate
    cv2.imshow('Imagine Blurată', imagine_blurata)
    cv2.imshow('Un Filtru Static', imagine_median)
    cv2.imshow('Filtru Derivat de Ordin 1', imagine_derivativa)
    cv2.imshow('Filtru Derivat de Ordin 2', imagine_laplaciana)

    # 3. Convoluție cu kernel1
    kernel1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    img_convolutie1 = cv2.filter2D(img, -1, kernel1)
    cv2.imshow('Convoluție cu Kernel1', img_convolutie1)

    # 4. Modificarea filtrului și o nouă convoluție
    # Schimbarea componentelor filtrului
    kernel2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    img_convolutie2 = cv2.filter2D(img, -1, kernel2)
    cv2.imshow('Convoluție cu Kernel2', img_convolutie2)

    # Așteaptă apăsarea unei taste și închide toate ferestrele
    cv2.waitKey(0)
    cv2.destroyAllWindows()
