import cv2
import numpy as np
from skimage.util import random_noise
import matplotlib.pyplot as plt
from scipy import signal
from skimage import filters
from PIL import Image, ImageFilter
path = r'C:\Users\nicuuu\PycharmProjects\pythonProject\image-refactoring\FourthLab\butterfly.jpeg'
img = cv2.imread(path,0)
