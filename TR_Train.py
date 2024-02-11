__author__ = 'mkv-aql'
import TR_Dataset as dataset
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
#from tensorflow import set_random_seed #tf version 1 compatibility, tf version 2 uses tf.random.set_seed
#set_random_seed(2)
tf.random.set_seed(2)

#input data path
classes = os.listdir('C:/Users/AGAM MUHAJIR/Desktop/Thiago_Rateke_Dataset/GT_1_all/')
num_classes = len(classes)

#print(classes)
# print(num_classes)

batch_size = 3 # will afffect RAM usage, usually 32 (lower for smaller RAM)
validation_size = 0.2 # 20% of the data will automatically be used for validation
img_size = 128 #Resize input images
num_channels = 3 #RGB channels
train_path = 'C:/Users/AGAM MUHAJIR/Desktop/Thiago_Rateke_Dataset/GT_1_all/'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size = validation_size)

# Showing progress
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


# Tensorflow session
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# Labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension = 1)


##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

# Define crate weights and biases functions
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

# Define CNN layer
def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    # Define the weights that will be trained using create_weights function
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # Define the biases using create_biases function
    biases = create_biases(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    # Add biases to the results of convolution
    layer += biases

    # Use pooling to downsample the image resolution
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Apply ReLU activation function
    layer = tf.nn.relu(layer)

    return layer

def create_flatten_layer(layer):
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # Calculate the number of features
    num_features = layer_shape[1:4].num_elements()

    # Flatten the layer
    layer = tf.reshape(layer, [-1, num_features])

    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # Define trainable weights and biases
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    # Fully connected layer takes input x and produces wx+b. This are matrices therefore using matmul from TF
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Achieving probabily distribution using Softmax for each class. Softmax function normalizes the output of the fully connected layer.
    # Adam optimizer is used to minimize the cost function, by updating the weights and biases in the input training data
layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
layer_conv3 = create_convolutional_layer(input=layer_conv2, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv3, num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input = layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs = fc_layer_size, user_relu = True)
layer_fc2 = create_fc_layer(input = layer_fc1, num_inputs=fc_layer_size, num_outputs = num_classes, user_relu = False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer()) #Running the training

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training epoch {0} --- Training accuracy: {1:>6.1%}, Validation accuracy: {2:>6.1%}, Validation loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

#Saving the model and also creating a folder for it
saver = tf.train.Saver() #saving the model
model_dir = 'C:/Users/AGAM MUHAJIR/Desktop/Thiago_Rateke_Dataset/Trained_Models/'
model_name = 'roadsurface-model'

#Create the directory if it does not exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)



def train(num_iteration):
    global total_iterations

    for i in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_btach = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict = feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            # Save the model. You can also include the epoch in the filename.
            saver.save(session, os.path.join(model_dir, model_name), global_step=epoch)
            # saver.save(session, './roadsurface-model')

    total_iterations += num_iteration

train(num_iteration = 60000)

print('\a')