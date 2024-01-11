__author__ = 'mkv-aql'

import os

# Import packages
# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile

from object_detection.builders import model_builder
from six.moves.urllib.request import urlopen
from six import BytesIO
from object_detection.utils import visualization_utils as viz_utils, config_util
from object_detection.utils import label_map_util

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

#Load model
model_dir = 'Models/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/saved_model'
detector = tf.saved_model.load(model_dir)
pipeline_config = 'Models/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/pipeline.config'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

# Function to load an image
#def load_image_into_numpy_array(path):

#    return np.array(Image.open(path))

def load_image_into_numpy_array(path):
    # Load image and convert to uint8
    image = np.array(Image.open(path))
    return image.astype(np.uint8)

# Load the downloaded and extracted model
model_path = 'Models/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/saved_model'
detect_fn = tf.saved_model.load(model_path)

# Load an image
image_path = 'Images/Hamburg4.JPG'
image_np = load_image_into_numpy_array(image_path)

# Run object detection
#input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

# Visualize the results
label_id_offset = 1
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes']+label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.30,
    agnostic_mode=False)

plt.figure()
plt.imshow(image_np_with_detections)
plt.show()