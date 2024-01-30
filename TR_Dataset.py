__author__ = 'mkv-aql'

import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


#Define ROI (Region of interest) and data augmentation
def adjust_gamma(image):
    gamma = 0.5
    invGamma = 1.0 / gamma #invert gamma value
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image,  table)

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255 #set all pixels in v to 255 if passes lim, prevents overflow in an 8-bit image format
    v[v <= lim] += value #increase v with 'value' amount if less or equal to limit

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


# ROI definition is made for each image when loading the inbput data
for fields in classes:
    index = classes.index(fields)
    print('Reading {} files (Index: {})'.format(fields, index))
    path = os.path.join(train_path, fields, '*g')
    files = glob.glob(path)
    for fl in files:
        image = cv2.imread(fl)

#         Region of Interest (ROI)
        height, width = image.shape[:2]
        newHeight = int(round(height / 2))
        image = image[newHeight-5:height-50, 0:width]

        brght_img = increase_brightness(image,value = 150)
        shaded_img = adjust_gamma(image)

        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        
