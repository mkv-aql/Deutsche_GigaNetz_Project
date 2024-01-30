__author__ = 'mkv-aql'
import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

classes = os.listdir('C:/Users/AGAM MUHAJIR/Desktop/Thiago_Rateke_Dataset/GT_1/')
num_classes = len(classes)

#print(classes)
# print(num_classes)

batch_size = 32
validation_size = 0.2
img_size = 128
num_channels = 3
train_path = 'C:/Users/AGAM MUHAJIR/Desktop/Thiago_Rateke_Dataset/GT_1/'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size = validation_size)


