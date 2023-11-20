import cv2
import numpy as np

# Citirea imaginii
img1 = cv2.imread('bmw4.jpg', cv2.IMREAD_GRAYSCALE)  # Convertirea la nivel de gri pentru cerintele ulterioare

# Verifică dacă imaginea a fost încărcată cu succes
if img1 is None:
    print("Error: Unable to load image.")
else:
    # 2.b. Transformări geometrice
    # Translație pe orizontală cu 30 de pixeli
    translatie_matrice = np.float32([[1, 0, 30], [0, 1, 0]])
    img_translatie = cv2.warpAffine(img1, translatie_matrice, (img1.shape[1], img1.shape[0]))

    # Rotire cu 30 de grade
    rotatie_matrice = cv2.getRotationMatrix2D((img1.shape[1] // 2, img1.shape[0] // 2), 30, 1)
    img_rotatie = cv2.warpAffine(img1, rotatie_matrice, (img1.shape[1], img1.shape[0]))

    # Transformare afină la alegere (exemplu: scalare)
    transformare_afina_matrice = np.float32([[0.5, 0, 0], [0, 1.5, 0]])
    img_transformare_afina = cv2.warpAffine(img1, transformare_afina_matrice, (img1.shape[1], img1.shape[0]))

    # Afișare imagini pentru transformările geometrice
    cv2.imshow('Imagine Originală', img1)
    cv2.imshow('Translație Orizontală', img_translatie)
    cv2.imshow('Rotire', img_rotatie)
    cv2.imshow('Transformare Afină', img_transformare_afina)

    # 2.c. Binarizare și calcul de descriptori geometrici
    _, img_binarizata = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)

    # Verifică dacă există contururi
    contur, _ = cv2.findContours(img_binarizata, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contur:
        # Aria obiectelor
        aria = cv2.contourArea(contur[0])

        # Perimetrele obiectelor
        perimetru = cv2.arcLength(contur[0], True)

        # Momentele Hu
        momente_hu = cv2.HuMoments(cv2.moments(contur[0])).flatten()

        # Elongația obiectelor (cu verificare pentru minim 5 puncte pentru a potrivii elipsa)
        if len(contur[0]) >= 5:
            _, _, elogatie = cv2.fitEllipse(contur[0])
            print(f'Aria: {aria}, Perimetru: {perimetru}, Momente Hu: {momente_hu}, Elongație: {elogatie}')
        else:
            print("Nu există suficiente puncte pentru a potrivi elipsa.")
    else:
        print("Nu s-au găsit contururi în imaginea binarizată.")

    # 3.a Umplere găuri și afișare imagine
    img_binarizata_umpluta = img_binarizata.copy()
    cv2.drawContours(img_binarizata_umpluta, [contur[0]], 0, 255, thickness=cv2.FILLED)
    cv2.imshow('Imagine Binarizată Umplută', img_binarizata_umpluta)

    # 3.b Reflexie față de orizontală și afișare
    img_reflexie_orizontala = cv2.flip(img_binarizata_umpluta, 1)
    cv2.imshow('Reflexie Față de Orizontală', img_reflexie_orizontala)

    # Așteaptă apăsarea unei taste și închide ferestrele
    cv2.waitKey(0)
    cv2.destroyAllWindows()
