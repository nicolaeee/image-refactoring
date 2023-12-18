import numpy as np
import cv2
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import random


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array, mode='gaussian', clip=True)


def blur_filter(img_array: ndarray):
    # blur the image
    return cv2.blur(img_array, (8, 8))

# for IAM dataset

def reduce_line_thickness(image: ndarray):
    kernel = np.ones((4,4), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def random_stretch(img: ndarray):
    stretch = (random.random() - 0.5)  # -0.5 .. +0.5
    wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
    img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5
    return img


aug_dir = "path/to/new_aug_lines/"
folder_dir = "path/to/data/lines/"


def return_file_names():
    gtTexts = []
    fileNames = []
    with open('data/line.txt', 'r') as f:
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')  ## remove the space and split with ' '
            # assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            # print(fileNameSplit)
            fileName = fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0]

            # GT text are columns starting at 10
            gtText_list = lineSplit[9].split('|')
            gtText = ' '.join(gtText_list)
            # put sample into list
            gtTexts.append(gtText)
            fileNames.append(fileName)
    print(fileNames[0].split('/'))
    return gtTexts, fileNames


def write_text(paths, gtTexts):
    with open('data/line_new.txt', 'w') as f:
        for i in range(len(paths)):
            text = gtTexts[i]
            t = text.split(' ')
            t = '|'.join(t)
            s = paths[i] + " x x x x x x x x " + t
            f.write(s + '\n')
    print("Write Sucessfully")


def create_aug_images():
    gtTexts, fileNames = return_file_names()
    image_paths = []
    image_texts = []
    # dictionary of the transformations functions we defined earlier
    available_transformations = [
        reduce_line_thickness,
        random_noise,
        blur_filter,
        random_stretch
    ]
    # all random numbers generated in a list to choose transformations
    choice = [random.randint(0, len(available_transformations) - 1) for p in range(0, len(gtTexts))]  # between a <= N <= b.
    print(choice)
    for i in range(0, len(gtTexts)):
        # read the image
        img_to_transform = sk.io.imread(folder_dir+fileNames[i]+".png")
        transformed_image = available_transformations[choice[i]](img_to_transform)
        img_name=aug_dir+"iam_aug_"+str(i)+".png"
        sk.io.imsave(img_name, transformed_image)
        image_paths.append("iam_aug_"+str(i)+".png")
        image_texts.append(gtTexts[i])
    write_text(image_paths, image_texts)