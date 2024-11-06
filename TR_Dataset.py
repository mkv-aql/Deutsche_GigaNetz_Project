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



# load training data
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Reading training images')

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
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            brght_img = cv2.resize(brght_img, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            brght_img = brght_img.astype(np.float32)
            brght_img = np.multiply(brght_img, 1.0 / 255.0)

            shaded_img = cv2.resize(shaded_img, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            shaded_img = shaded_img.astype(np.float32)
            shaded_img = np.multiply(brght_img, 1.0 / 255.0)

    #         balancing input images, because there are more asphalt and fewer paved and unpaved
            if index == 0: #asphalt
                images.append(image)
                images.append(brght_img)
                images.append(shaded_img)

                label = np.zeros(len(classes))
                label[index] = 1.0

                labels.append(label)
                labels.append(label)
                labels.append(label)

                flbase = os.path.basename(fl)

                img_names.append(flbase)
                img_names.append(flbase)
                img_names.append(flbase)

                cls.append(fields)
                cls.append(fields)
                cls.append(fields)

            elif index == 1: #paved
                for i in range(3):
                    images.append(image)
                    images.append(brght_img)
                    images.append(shaded_img)

                    label = np.zeros(len(classes))
                    label[index] = 1.0

                    labels.append(label)
                    labels.append(label)
                    labels.append(label)

                    flbase = os.path.basename(fl)

                    img_names.append(flbase)
                    img_names.append(flbase)
                    img_names.append(flbase)

                    cls.append(fields)
                    cls.append(fields)
                    cls.append(fields)


            elif index == 2: #unpaved
                for i in range(6):
                    images.append(image)
                    images.append(brght_img)
                    images.append(shaded_img)

                    label = np.zeros(len(classes))
                    label[index] = 1.0

                    labels.append(label)
                    labels.append(label)
                    labels.append(label)

                    flbase = os.path.basename(fl)

                    img_names.append(flbase)
                    img_names.append(flbase)
                    img_names.append(flbase)

                    cls.append(fields)
                    cls.append(fields)
                    cls.append(fields)

    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(labels)
    cls = np.array(cls)

    return images, labels, img_names, cls

# Define DataSet class, to handle a single dataset
class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        #Reshape the images into 4D array
        # [image_number, height, width, channel]
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], images.shape[3])

        #Convert the input images into tensors
        #self._images = tf.convert_to_tensor(images, dtype = tf.float32)
        self._images = images

        self._labels = labels
        self._img_names = img_names
        self._cls = cls

        self._epochs_done = 0
        self._index_in_epoch = 0

    #Define images, labels, img_names, cls as properties
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        # return next batch size examples from current dataset
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            #After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size

            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):

    #Turn them data into arrays
    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    #Shuffle the data arrays
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    #Turn the validation_size type into an integer, just in case it is a decimal
        #images.shape[0] will return height size in pixels
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    #Training data
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    #Validation data
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    # Simple container or structure to hold training and validation sets called from class DataSet
        # Create DataSets instance so that it can be called from the class DataSet (data_sets = DataSet())
    class DataSets(object):
        pass

    data_sets = DataSets()

    #Assign the training and validation data to the DataSets class instances / call the DataSet class object instance twice (train and valid)
    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets

