import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import cv2
from utils import visualization_utils as vis_util
import time

def create_category_index(label_path='labels.txt'):
	"""
	To create dictionary of label map
	Parameters
	----------
	label_path : string, optional
		Path to labelmap.txt. The default is 'coco_ssd_mobilenet/labelmap.txt'.
	Returns
	-------
	category_index : dict
		nested dictionary of labels.
	"""
	f = open(label_path)
	category_index = {}
	for i, val in enumerate(f):
		if i != 0:
		    val = val[:-1]
		    if val != '???':
		        category_index.update({(i-1): {'id': (i-1), 'name': val}})

	f.close()
	return category_index

def get_output_dict(image, interpreter, output_details, nms=True, iou_thresh=0.5, score_thresh=0.6):
	output_dict = {
                   'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                   'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                   'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                   'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                   }
	output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
	return output_dict

def make_and_show_inference(img, interpreter, input_details, output_details, nms=False, score_thresh=0.5, iou_thresh=0.5):
	#start_time = time.perf_counter()
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
	img_rgb = img_rgb.reshape([1, 300, 300, 3])
	interpreter.set_tensor(input_details[0]['index'], img_rgb)
	interpreter.invoke()
	output_dict = get_output_dict(img_rgb, interpreter, output_details, nms, iou_thresh, score_thresh)
	for idx, scores in enumerate(output_dict['detection_scores']):
		if scores > 1.00:
			del output_dict['detection_boxes'][idx]
			del output_dict['detection_classes'][idx]
			del output_dict['detection_scores'][idx]
		else:
			break
	if nms:
		output_dict = apply_nms(output_dict, iou_thresh, score_thresh)
	#print(output_dict)

	vis_util.visualize_boxes_and_labels_on_image_array(
    img,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    min_score_thresh=score_thresh,
    line_thickness=3)

	return output_dict

def apply_nms(output_dict, iou_thresh=0.5, score_thresh=0.6):
	q = 90 # no of classes
	num = int(output_dict['num_detections'])
	boxes = np.zeros([1, num, q, 4])
	scores = np.zeros([1, num, q])

	for i in range(num):
		# indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
		boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
		scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
	nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
		                                         scores=scores,
		                                         max_output_size_per_class=num,
		                                         max_total_size=num,
		                                         iou_threshold=iou_thresh,
		                                         score_threshold=score_thresh,
		                                         pad_per_class=False,
		                                         clip_boxes=False)
	valid = nmsd.valid_detections[0].numpy()
	output_dict = {
		           'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
		           'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
		           'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
		           }
	return output_dict

tflite_path="/home/elid/siewwen/Object_detection_with_augmentation/export_graphs/200EDE_40aug_newaug_445val/200EDE_40aug_newaug_445val.tflite" # CHANGE !!!!!!!!!!!!

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(tflite_path)
interpreter.allocate_tensors()
#interpreter2.allocate_tensors()
category_index = create_category_index()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
tf.compat.v1.enable_eager_execution()

cap = cv2.VideoCapture('/home/elid/siewwen/test_video/testVid1.avi')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/elid/siewwen/test_video/testVid_1_200EDE.avi',fourcc, 30, size)

counter = 0
while(True):
	ret, img = cap.read()
	counter += 1
	# if (ret and (counter >2200)):
	if ret:
		make_and_show_inference(img, interpreter, input_details, output_details)
		# cv2.imshow("image", img)
		out.write(img)
		# if cv2.waitKey(0) & 0xFF == ord('q'):
		# 	cap.release()
		# 	out.release()
		# 	break
	elif(not ret):
		break

import csv
def actual_boxes(image_dir):
	# csv_input = "/home/elid/siewwen/datasets/HDA_Dataset_V1.3/HDA_Dataset_V1.3/hda_image_sequences_matlab/tmp.csv" #CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	csv_input = "/home/elid/siewwen/Object_detection_with_augmentation/output2.csv"
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
				# print(box_actual)
	return frame_width, frame_height, boxes_actual

# import glob
# allfiles = glob.glob("/home/elid/siewwen/Object_detection_with_augmentation/final_val/images/*.jpg")
# allfiles.sort()
# for file in allfiles:
# 	img = cv2.imread(file)
#
# 	# get the actual boxes
# 	# width, height, boxes_act = actual_boxes(file)
# 	#
# 	# for box_act in boxes_act:
# 	# 	cv2.rectangle(img, (box_act[1], box_act[0]), (box_act[3], box_act[2]),(255, 0, 0), 3)
# 	# if (width == None) or (height == None):
# 	# 	continue
# 	make_and_show_inference(img, interpreter, input_details, output_details)
# 	cv2.imshow("image", cv2.resize(img, (600,600)))
# 	if cv2.waitKey(0) & 0xFF == ord('q'):
# 		break


#cap.release()
cv2.destroyAllWindows()
