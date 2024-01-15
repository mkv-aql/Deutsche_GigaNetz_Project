__author__ = 'mkv-aql'

from PIL import Image
import numpy as np
import tensorflow as tf
import keras_cv

# Load and preprocess the image
image = Image.open('Images/Hamburg3.jpg')
image = image.resize((224, 224))
input_data = np.array(image)[np.newaxis, ...]  # Add batch dimension

# Load YOLOV8 model
model = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone")

# Perform object detection
output = model(input_data)


# This depends on your model's specific output format
boxes = output['boxes']
scores = output['scores']
classes = output['classes']

# Apply a threshold to filter out low-confidence detections
threshold = 0.5
indices = [i for i, score in enumerate(scores) if score > threshold]

# Draw the bounding boxes on the image
for i in indices:
    box = boxes[i]
    class_id = classes[i]
    score = scores[i]

import matplotlib.pyplot as plt

# Convert the PIL image to a format that can be displayed by matplotlib
image_with_boxes = np.array(image)  # If 'image' is a PIL Image object

plt.figure(figsize=(12, 8))
plt.imshow(image_with_boxes)
plt.axis('off')  # Turn off axis numbers
plt.show()