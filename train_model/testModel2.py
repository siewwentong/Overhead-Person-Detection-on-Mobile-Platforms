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
import glob

import cv2
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# function to get the IOU
# boxA and box B are the xmin xmax ymin ymax for the detected box and the actual box
def compute_IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	y_min = max(boxA[0], boxB[0])
	x_min = max(boxA[1], boxB[1])
	y_max = min(boxA[2], boxB[2])
	x_max = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

# funtion to get the actual boxes for the image passed in
def actual_boxes(image_dir):
	# csv_input = "/home/elid/siewwen/datasets/HDA_Dataset_V1.3/HDA_Dataset_V1.3/hda_image_sequences_matlab/tmp.csv" #CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	csv_input = "/home/elid/siewwen/Object_Detection_Trainer/out_rotated.csv"
	index = image_dir.rfind('/')
	image_name = image_dir[index+1:]
	boxes_actual = []
	frame_width = None
	frame_height = None

	# open csv and read the file row by row
	with open(csv_input, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			if row[0] == image_name:
				x_min = int(row[3])
				y_min = int(row[4])
				x_max = int(row[5])
				y_max = int(row[6])
				frame_width = int(row[2])
				frame_height = int(row[1])
				box_actual = [y_min, x_min, y_max, x_max]
				boxes_actual.append(box_actual)
	return frame_width, frame_height, boxes_actual


allfiles = glob.glob("/home/elid/siewwen/Object_Detection_Trainer/rotated_images/*.jpg") #CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# allfiles = glob.glob("/home/elid/siewwen/datasets/HDA_Dataset_V1.3/HDA_Dataset_V1.3/hda_image_sequences_matlab/filtered/*.jpg")
# pngfiles = glob.glob("/home/elid/siewwen/Object_Detection_Trainer/images/*.png")
# for pngfile in pngfiles:
# 	allfiles.append(pngfile )
allfiles.sort()

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  "/home/elid/siewwen/Object_detection_with_augmentation/export_graphs/wo_new_aug_hue2/frozen_inference_graph.pb" # CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/elid/siewwen/Object_detection_with_augmentation/labels.pbtxt"

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

confidence_arr = np.arange(0.05, 1, 0.05)
confidence_arr = [0.5]
TPrate = []
FPrate = []

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		for confidence in confidence_arr:
			# for each confidence threshold, reset the value
			false_positive = 0
			true_positive = 0
			total_positive = 0
			number_im = 0
			for file in allfiles:
				# print(file)
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np = cv2.imread(file)
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

				# get the actual boxes
				width, height, boxes_act = actual_boxes(file)
				for box_act in boxes_act:
					cv2.rectangle(image_np, (box_act[1], box_act[0]), (box_act[3], box_act[2]),(255, 0, 0), 3)
				if (width == None) or (height == None):
					continue
				detection = 0
				total_positive = total_positive + len(boxes_act)

				for index in range(len(scores[0])):
					if scores[0][index] >= confidence and scores[0][index] <= 1:
						detection = detection + 1;
						# change the coordinate to the size of the image
						box_detect = [boxes[0][index][0]*height, boxes[0][index][1]*width, boxes[0][index][2]*height, boxes[0][index][3]*width]
						iou_box = []
						for box_act in boxes_act:
							iou_box.append(compute_IOU(box_act, box_detect))
						# if there's no more boxes to compare, all leftover detecte boxes are false positive
						if len(iou_box) > 0:
							max_iou = max(iou_box)
							if max_iou > 0.5:
								boxes_act.pop(iou_box.index(max_iou))
								true_positive = true_positive + 1
							else:
								false_positive = false_positive + 1
						else:
							false_positive = false_positive + 1

				number_im = number_im + 1;

				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
				    image_np,
				    np.squeeze(boxes),
				    np.squeeze(classes).astype(np.int32),
				    np.squeeze(scores),
				    category_index,
				    use_normalized_coordinates=True,
				    line_thickness=2)

				cv2.imshow('object detection', cv2.resize(image_np, (600,600)))
				cv2.waitKey(1)

			print(confidence)
			TPrate.append(true_positive/total_positive)
			print(true_positive/total_positive)
			FPrate.append(false_positive/number_im)
			print(false_positive/number_im)

# plot graph
plt.plot(FPrate, TPrate)
plt.xlabel("false positive per image")
plt.ylabel("true positive rate")
plt.title("ROC curve")
# plt.savefig("/home/elid/siewwen/Object_Detection_Trainer/ROC_graphs/with_cafeteria_on_val_img.png") # CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
