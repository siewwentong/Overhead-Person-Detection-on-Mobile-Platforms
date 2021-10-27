from create_synthetic_images_background_subtraction import Dataset, train_background_model, create_dataset_array, fillhole
import numpy as np
import cv2
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_list", default = "all_datasets.txt")
args = parser.parse_args()

with open(args.dataset_list, 'r') as txt:
	dataset_lists = txt.readlines()
	dataset_lists.sort()
	# getting the datasets name
	datasets = []
	for dataset_list in dataset_lists:
		name = dataset_list[:-1] +"/"
		# create a new Dataset object
		new_dataset = Dataset(name,create_dataset_array(dataset_list[:-1]))
		# train the background model
		train_background_model(dataset_list[:-1], new_dataset.background_model, new_dataset.history)
		datasets.append(new_dataset)
		# print(new_dataset.name)
		print(len(new_dataset.img_list))

	for dataset in datasets:
		for image_name in dataset.img_list:
			image = cv2.imread(dataset.name+image_name)
			# create the file structure
			annotation = ET.Element('annotation')
			filename = ET.SubElement(annotation, 'filename')
			filename.text = image_name
			size = ET.SubElement(annotation, 'size')
			width = ET.SubElement(size, 'width')
			height = ET.SubElement(size, 'height')
			width.text = str(image.shape[1])
			height.text = str(image.shape[0])
			# apply background model to get foreground mask
			foreground_mask = dataset.background_model.apply(image)
			# apply thresholding to get binary image (if > 200, pixel = 1, else 0)
			ret, foreground_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
			'''
			to understand closing and opening, see:
			https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
			'''
			# applying closing to enclose holes that are not closed
			# foreground_mask = foreground_mask[50:200]

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
			new_im = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
			# applying opening to remove noises
			new_im = cv2.morphologyEx(new_im, cv2.MORPH_OPEN, kernel)
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
			# applying closing to enclose holes that are not closed
			new_im = cv2.morphologyEx(new_im, cv2.MORPH_CLOSE, kernel)
			# applying closing to enclose holes that are not closed
			new_im = cv2.morphologyEx(new_im, cv2.MORPH_CLOSE, kernel)
			# applying closing to enclose holes that are not closed
			new_im = cv2.morphologyEx(new_im, cv2.MORPH_CLOSE, kernel)

			# fill holes that are onclosed
			new_im = fillhole(new_im)

			num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_im, connectivity=8)
			print(image_name)
			bbox = []
			for stat in stats:
				left, top, width, height, area = stat
				if area > 4000 and area < 80000:
					xmin = left
					ymin = top
					xmax = xmin + width
					ymax = ymin + height
					if (ymin < 100 or ymax > 400 or xmax == image.shape[1]):
						continue
					bbox.append([xmin, ymin, xmax, ymax])

			for box in bbox:
				xmin, ymin, xmax, ymax = box
				# print(xmin, ymin, xmax, ymax)
				object = ET.SubElement(annotation, 'object')
				name = ET.SubElement(object, 'name')
				name.text = 'person'
				pose = ET.SubElement(object, 'pose')
				pose.text = 'Unspecified'
				truncated = ET.SubElement(object, 'truncated')
				truncated.text = '1'
				difficult = ET.SubElement(object, 'difficult')
				difficult.text = '0'
				bndbox = ET.SubElement(object, 'bndbox')
				minx = ET.SubElement(bndbox, 'xmin')
				minx.text = str(xmin)
				miny = ET.SubElement(bndbox, 'ymin')
				miny.text = str(ymin)
				maxx = ET.SubElement(bndbox, 'xmax')
				maxx.text = str(xmax)
				maxy = ET.SubElement(bndbox, 'ymax')
				maxy.text = str(ymax)
				cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),3)
			# cv2.imshow("image", cv2.resize(image, (640,480)))
			# cv2.waitKey(0)
			# cv2.imshow("mask", new_im)
			# cv2.waitKey(0)

			if len(bbox) == 1:
				# create a new XML file with the results
				mydata = ET.tostring(annotation, encoding="unicode")
				xml_name = dataset.name + image_name[:-3] + "xml"
				print("creating xml file: " + xml_name)
				myfile = open(xml_name, "w")
				myfile.write(mydata)
