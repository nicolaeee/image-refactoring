import cv2
import numpy as np
from matplotlib import pyplot as plt

# Citirea imaginii
img = cv2.imread('pix.jpeg')
img2 = cv2.imread('pix.jpeg', cv2.IMREAD_GRAYSCALE)

# Aplicarea unui filtru median de ordinul 1
img_filtered = cv2.medianBlur(img, 3)  # 3 este dimensiunea kernel-ului, puteți ajusta la nevoie
median = cv2.medianBlur(img, 5)

# Aplicarea filtrului Laplacian
laplacian = cv2.Laplacian(img2, cv2.CV_64F)
laplacian_rgb = cv2.cvtColor(np.abs(laplacian).astype(np.uint8), cv2.COLOR_BGR2RGB)

# Conversie înapoi la uint8 pentru a evita eroarea de tip în calculul PSNR
laplacian_uint8 = np.abs(laplacian).astype(np.uint8)

# Calcularea raportului de semnal-zgomot (SNR) între imaginea originală și cea filtrată
snr_original = cv2.PSNR(img, median)
snr_laplacian = cv2.PSNR(img2, laplacian_uint8)

# Convertirea imaginilor la formatul RGB (matplotlib folosește acest format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_filtered_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)

# Crearea subplot-ului 2x2 și afișarea imaginii originale pe poziția (1, 1)
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Imaginea de la "pix.jpeg"')

# Afisarea imaginii filtrate pe poziția (2, 1)
plt.subplot(2, 2, 2)
plt.imshow(median)
plt.title('Filtru Ordinul 1')

# Afisarea imaginii filtrate cu filtrul Laplacian pe poziția (2, 2, 3)
plt.subplot(2, 2, 3)
plt.imshow(laplacian)

# Afisarea rapoartelor de semnal-zgomot
plt.subplot(2, 2, 3)
plt.text(0.5, 0.5, f'SNR Original: {snr_original:.2f} dB\nSNR Laplacian: {snr_laplacian:.2f} dB', ha='center', va='center', fontsize=12)
plt.axis('off')

# Afisarea subplot-urilor
plt.show()


