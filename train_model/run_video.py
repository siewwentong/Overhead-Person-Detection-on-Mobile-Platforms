import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--vid_file", help="video file", required=True)
parser.add_argument("--exported_graph_file", help="exported graph dir", required=True)
parser.add_argument("--label_file", help="pbtxt file path", required=True)
parser.add_argument("--num_classes", help="number of classes", default=5)
args = parser.parse_args()

#cap = cv2.VideoCapture('/home/elid/google-coral/armani condo/20190809/5a-57-64-3a-cf-9e/channel01/1_EXIT_2_20190809-171908.AVI')
cap = cv2.VideoCapture(args.vid_file)
sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = args.exported_graph_file


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = args.label_file

NUM_CLASSES = int(args.num_classes)


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
	    ret, image_np = cap.read()
	    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	    image_np_expanded = np.expand_dims(image_np, axis=0)
	    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	    # Each box represents a part of the image where a particular object was detected.
	    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	    # Each score represent how level of confidence for each of the objects.
	    # Score is shown on the result image, together with the class label.
	    scores = detection_graph.get_tensor_by_name('detection_scores:0')
	    classes = detection_graph.get_tensor_by_name('detection_classes:0')
	    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	    # Actual detection.
	    (boxes, scores, classes, num_detections) = sess.run(
	      [boxes, scores, classes, num_detections],
	      feed_dict={image_tensor: image_np_expanded})
	    # Visualization of the results of a detection.
	    vis_util.visualize_boxes_and_labels_on_image_array(
	        image_np,
	        np.squeeze(boxes),
	        np.squeeze(classes).astype(np.int32),
	        np.squeeze(scores),
	        category_index,
	        use_normalized_coordinates=True,
	        line_thickness=8)

	    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	      cv2.destroyAllWindows()
	      break
