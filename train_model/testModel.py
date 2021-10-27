import numpy as np
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

cap = cv2.VideoCapture('/home/elid/siewwen/test_video/testVideo_1.avi')
# save output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/elid/siewwen/test_video/textVideo_1_wo_new_aug_hue2_blur.avi',fourcc, float(30/4), size)

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/elid/siewwen/Object_detection_with_augmentation/export_graphs/with_TVMPC_10_augmentations_wo_depthwise_smaller_regularizer_150000/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/elid/siewwen/elid_eye_labels.pbtxt'

NUM_CLASSES = 1 #TO_BE_CONFIGURED

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

confidence = 1

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
			while True:
				ret, image_np = cap.read()
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				#image_np_expanded = np.expand_dims(image_np, axis=0)
				image_np_expanded = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
				image_np_expanded = cv2.resize(image_np_expanded, (300, 300), cv2.INTER_AREA)
				image_np_expanded = image_np_expanded.reshape([1, 300, 300, 3])
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
				    line_thickness=2)

				cv2.imshow('object detection', image_np)
				out.write(image_np)
				if cv2.waitKey(0) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					cap.release()
					out.release()
					break
