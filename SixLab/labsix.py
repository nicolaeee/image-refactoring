import cv2
import numpy as np
from matplotlib import pyplot as plt

# Citirea imaginii
img = cv2.imread('download.jpeg')

# Aplicarea filtrului Gaussian
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Calcularea raportului de semnal-zgomot (SNR)
snr = cv2.PSNR(img, img_gaussian)

# Convertirea imaginilor la formatul RGB (matplotlib folosește acest format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gaussian_rgb = cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB)

# Crearea subplot-ului 2x2 și afișarea imaginii originale pe poziția (1, 1)
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Imaginea Originală')

# Afisarea imaginii filtrate pe poziția (2, 2)
plt.subplot(2, 2, 2)
plt.imshow(img_gaussian_rgb)
plt.title('Imaginea cu Filtru Gaussian')

# Afisarea raportului de semnal-zgomot
plt.subplot(2, 2, 3)
plt.text(0.5, 0.5, f'Semnal-zgomot: {snr:.2f} dB', ha='center', va='center', fontsize=12)
plt.axis('off')

# Aici scrie codul pentru binarizarea imaginei și aplică transformarea morfologică de tip deschidere doar sublop ului 2, 2, 3
plt.subplot(2, 2, 3)
# Afisarea subplot-urilor
plt.show()
