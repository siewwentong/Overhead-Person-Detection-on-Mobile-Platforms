'''
This code generates augmented images according to operations in augmentation_list, writes augmented annotations in csv_output, and saves augmented images in output_dir
Written by: Tong Siew Wen (January 2021)
...
How to use this code:
- python3 augmentation_per_dataset.py --dataset <dataset_name> --csv_output <xxx.csv> (add on other parameters if needed)
-
...
What is required:
- a list (textfile) of images to be augmented (input for "--image_list", default = "to_augment.txt")
- a list (textfile) of data augmentations combinations (aka operations) to be implemented on the images
	- Operations will be eliminated if: (as these combination will cause the augmented imgaes to be compromised)
		- Hue, saturation, brightness and contrast (all four of them) are in the operation
		- there are more than four data augmentation, and hue + saturation OR brightness + contrast is in the operation
...
To take note:
- name of xml file should be the same as the image name (abc.jpg -> abc.xml)
'''

import cv2
import csv
import math
import xml.etree.ElementTree as ET
import colorsys
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import os
import glob
import random
import augmentations as aug
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--csv_output", default = "out_augmented.csv")
parser.add_argument("--image_list", default = "to_augment.txt")
parser.add_argument("--augmentation_list", default = "data_augmentation_combination.txt")
parser.add_argument("--dataset")
parser.add_argument("--image_dir", default = "images/")
parser.add_argument("--xml_dir", default = "images/")
parser.add_argument("--output_dir", default = "augmented_images/")
args = parser.parse_args()

def annotation_augmentation(operation_list, width, height, xmin, ymin, xmax, ymax, aspect_ratio):
	'''
	Function: assigns annotation augmentations
	...
	Input parameter:
	operation list = data augmentation combination (python list of strings)
	width = width of image (integer only)
	height = height of image (integer only)
	xmin = minimum x coordinate of bounding box (integer only)
	ymin = minimum y coordinate of bounding box (integer only)
	xmax = maximum x coordinate of bounding box (integer only)
	ymax = maximum y coordinate of bounding box (integer only)
	aspect_ratio = float that represents width:height (e.g aspect ratio = 1.5 -> width:height = 3:2)
	...
	Return:
	width = new width of image (integer only)
	height = new height of image (integer only)
	xmin = new minimum x coordinate of bounding box (integer only)
	ymin = new minimum y coordinate of bounding box (integer only)
	xmax = new maximum x coordinate of bounding box (integer only)
	ymax = new maximum y coordinate of bounding box (integer only)
	'''
	for operation_name in operation_list:
		if operation_name == "Rotation90": # augment annotation for rotation 90 degrees clockwise
			width, height, xmin, ymin, xmax, ymax = aug.rotate_box(width, height, xmin, ymin, xmax, ymax, 90)
		elif operation_name == "Rotation180": # augment annotation for rotation 180 degrees clockwise
			width, height, xmin, ymin, xmax, ymax = aug.rotate_box(width, height, xmin, ymin, xmax, ymax, 180)
		elif operation_name == "Rotation270": # augment annotation for rotation 90 degrees clockwise
			width, height, xmin, ymin, xmax, ymax = aug.rotate_box(width, height, xmin, ymin, xmax, ymax, 270)
		elif operation_name == "Flip_ver": # augment annotation for fliping vertically
			width, height, xmin, ymin, xmax, ymax = aug.flip_box(width, height, xmin, ymin, xmax, ymax, 0)
		elif operation_name == "Flip_hori": # augment annotation for fliping horizontally
			width, height, xmin, ymin, xmax, ymax = aug.flip_box(width, height, xmin, ymin, xmax, ymax, 1)
		elif operation_name == "Ratio": # augment annotation for changing aspect ratio
			width, height, xmin, ymin, xmax, ymax = aug.change_aspect_ratio_box(width, height, xmin, xmax, ymin, ymax, aspect_ratio)
	return width, height, xmin, ymin, xmax, ymax

def image_augmentation(operation_name, filename, filename_new, no_of_op, reduce, dataset):
	'''
	Function: assigns image augmentations
	...
	Input parameter:
	operation name = name of data augmentation to apply (string)
	filename = path to image to be augmented (string) (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (string) (relative to current working directory) -> e.g. save_dir/new_name.jpg
	no_of_op = number of data augmentations in the combination (extend of augmentation is reduced when no_of_op > 3) (integer ONLY)
	reduce = Boolean to indicate whether to reduce the extend of augmentation (boolean ONLY)
	dataset = the name of the dataset where this image is from (string) -> to take care of dataset that is sensitive to augmentations
	Return:
	aspect_ratio = float that represents width:height (e.g aspect ratio = 1.5 -> width:height = 3:2)
	^ these two are to be inputed for annotation augmentation (so that image and annotation are augmented with the same parameters)
	'''
	aspect_ratio = None
	if operation_name == "Rotation90": # rotate image 90 degree clockwise
		aug.rotate_image(filename, filename_new, cv2.ROTATE_90_CLOCKWISE)

	elif operation_name == "Rotation180": # rotate image 90 degree clockwise
		aug.rotate_image(filename, filename_new, cv2.ROTATE_180)

	elif operation_name == "Rotation270": # rotate image 90 degree clockwise
		aug.rotate_image(filename, filename_new, cv2.ROTATE_90_COUNTERCLOCKWISE)

	elif operation_name == "Flip_ver": # flip image vertically
		aug.flip_image(filename, filename_new, 0)

	elif operation_name == "Flip_hori": # flip image vertically
		aug.flip_image(filename, filename_new, 1)

	elif operation_name == "Hue1": # Shift hue of image (range 1)
		shift = random.randint(10,25) # shift between 0.1 to 0.25
		aug.change_hue(filename, filename_new, shift/100.0)

	elif operation_name == "Hue2": # Shift hue of image (range 2)
		shift = random.randint(30,50) # shift between 0.3 to 0.5
		aug.change_hue(filename, filename_new, shift/100.0)

	elif operation_name == "Hue3": # Shift hue of image (range 3)
		shift = random.randint(55,75) # shift between 0.55 to 0.75
		aug.change_hue(filename, filename_new, shift/100.0)

	elif operation_name == "Hue4": # Shift hue of image (range 4)
		shift = random.randint(80,90) # shift between 0.8 to 0.9
		aug.change_hue(filename, filename_new, shift/100.0)

	elif operation_name == "Saturation_in": # Increase saturation of image
		flag = True if (dataset == "synthetic_images" or no_of_op > 3 or reduce) else False
		# if the images are synthetic OR number of data augmentation per combination is more than 3
		# OR other scenerios that the extend of augmentation need to be reduced, reduce the extennd of augmentation
		shift = random.randint(12,15) if flag else random.randint(15,20)
		aug.change_saturation(filename, filename_new, shift/10.0) # shift between 1.2 to 2.0

	elif operation_name == "Saturation_de": # Decrease saturation of image
		flag = True if (dataset == "EDOD" or dataset == "synthetic_images" or no_of_op > 3 or reduce) else False
		# if the images are synthetic OR from EDOD dataset OR number of data augmentation per combination is more than 3
		# OR other scenerios that the extend of augmentation need to be reduced, reduce the extennd of augmentation
		shift = random.randint(6,8) if flag else random.randint(3,5)
		aug.change_saturation(filename, filename_new, shift/10.0) # shift between 0.3 to 0.8

	elif operation_name == "Brightness_in": # Increase brightness of image
		flag = True if (dataset != "EDOD") and (dataset == "synthetic_images" or no_of_op > 3) else False
		# if the images are synthetic OR number of data augmentation per combination is more than 3, reduce the extennd of augmentation
		# But if images are from EDOD dataset, do not reduce the extend of augmentation regardless
		shift = random.randint(12,15) if flag else random.randint(15,20)
		aug.change_brightness(filename, filename_new, shift/10.0) # shift between 1.2 to 2.0

	elif operation_name == "Brightness_de": # Decrease brightness of image
		flag = True if (dataset == "EDOD" or dataset == "synthetic_images" or no_of_op > 3) else False
		# if the images are synthetic OR from EDOD dataset OR number of data augmentation per combination is more than 3, reduce the extennd of augmentation
		shift = random.randint(6,8) if flag else random.randint(3,5)
		aug.change_brightness(filename, filename_new, shift/10.0) # shift between 0.3 to 0.8

	elif operation_name == "Contrast_in": # Increase contrast of image
		flag = True if (dataset == "synthetic_images" or no_of_op > 3) else False
		# if the images are synthetic OR number of data augmentation per combination is more than 3, reduce the extennd of augmentation
		shift = random.randint(12,15) if flag else random.randint(15,20)
		aug.change_contrast(filename, filename_new, shift/10.0)  # shift between 1.2 to 2.0

	elif operation_name == "Contrast_de": # Decrease contrast of image
		flag = True if (dataset == "EDOD" or dataset == "synthetic_images" or no_of_op > 3) else False
		# if the images are synthetic OR from EDOD dataset OR number of data augmentation per combination is more than 3, reduce the extennd of augmentation
		shift = random.randint(6,8) if flag else random.randint(3,5)
		aug.change_contrast(filename, filename_new, shift/10.0) # shift between 0.3 to 0.8

	elif operation_name == "Noise": # Add salt and peper noise to image
		flag = True if (dataset == "synthetic_images" or no_of_op > 3) else False
		# if the images are synthetic OR number of data augmentation per combination is more than 3, reduce the extennd of augmentation
		prob = random.randint(6,9) if flag else random.randint(10,15)
		aug.add_noise(filename, filename_new, prob/1000) # probablity between 0.006 to 0.015

	elif operation_name == "Blur": # Blur the image using Gaussian Blur
		# reduce the blur radius as the number of operation in combination increases (to prevent image to be compromised)
		radius = 12 - min(8, math.floor(no_of_op/2.0))
		aug.blur_image(filename, filename_new, radius/4.0) # maximum blur radius of 3.0

	elif operation_name == "Grid": # Apply grid mask to image
		aug.grid_mask(filename, filename_new)

	elif operation_name == "Ratio": # Change aspect ratio of image
		aspect_ratio = (random.randint(5, 20))/10.0 # aspect ratio can be between 0.5 to 2
		aug.change_aspect_ratio_image(filename, filename_new, aspect_ratio)

	return aspect_ratio

def create_dataset_array(dataset_list):
	'''
	Function: returns a list of images that is in the folder that correspond to that dataset
	...
	Input parameter:
	dataset = path to the textfile that consist all the images from that dataset (relative to the current working directory)
	'''
	with open(dataset_list, 'r') as txt_file:
		images = txt_file.readlines()
	img_list = []
	for image in images:
		img_list.append(image[:-1])
	return img_list

# main code
def main(args):

	# if output directory doesn't exists, create one
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	with open(args.augmentation_list,'r') as file:
		contents = file.readlines()
		Hue = ["Hue1", "Hue2", "Hue3", "Hue4"]
		Saturation = ["Saturation_in", "Saturation_de"]
		Brightness = ["Brightness_in", "Brightness_de"]
		Contrast = ["Contrast_in", "Contrast_de"]
		operations = []
		for content in contents:
			# create a list of data augmentation for each combination
			operation = content[:-1].split(" ")
			# check whether hue, saturation, brightness, contrast exists in the combination
			hv_hue = any(item in Hue for item in operation)
			hv_saturation = any(item in Saturation for item in operation)
			hv_brightness = any(item in Brightness for item in operation)
			hv_contrast = any(item in Contrast for item in operation)

			# if there are more than four data augmentations in the combination
			if len(operation) > 4:
				if (hv_hue and hv_saturation) or (hv_brightness and hv_contrast) :
					# if saturation and hue or contrast and brightness exists together, skip this operation
					continue
			else:
				# if saturation, hue, contrast and brightness exists together, skip this operation
				if (hv_hue and hv_saturation) and (hv_brightness and hv_contrast):
					continue

			# save combinations in a list (operations)
			operations.append(operation)
		# print(len(operations))
		# print(len(operations[-1]))
		# exit()

		with open(args.image_list, 'r') as ip:
			# getting the list of images to be augmented
			images = ip.readlines()
			with open(args.csv_output, mode='w', newline='') as output_file:
				writer = csv.writer(output_file)

				# Get a list of images that is in the folder that correspond to that dataset
				dataset = create_dataset_array("dataset_list/" + args.dataset + ".txt")

				img_list = []
				for image in images:
					# if image to be augmented belongs to that dataset, append it in img_list
					if (image[:-1] in dataset):
						img_list.append(image[:-1])

				# length = max(len(img_list), len(operations))
				length = len(img_list)*10

				for i in range(length):
					i_op = i # index for operation to use
					i_im = i # index for image to be augmented
					while i_op > len(operations)-1:
						# loop back to the first operation if the end was reached
						i_op -= len(operations)
					while i_im > len(img_list)-1:
						# loop back to the first image if the end was reached
						i_im -= len(img_list)
					filename = img_list[i_im]

					filename_new = None
					# creating the new filename (dataset name + original filename + all augmentations applied)
					for operation in operations[i_op]:
						if operations[i_op].index(operation) == 0:
							filename_new = args.output_dir + args.dataset + "_" + filename[:-4]
						filename_new = filename_new + "_" + operation
					filename_new = filename_new + ".jpg"
					# perform each operation on image
					print(operations[i_op])
					aspect_ratio = None

					for operation in operations[i_op]:
						if any(item in Hue for item in operation) and any(item in Saturation for item in operation):
							# if saturation and hue exists in combination, reduce the ectend of augmentation
							reduce = True
						else:
							reduce = False
						if operations[i_op].index(operation) == 0:
							# if original image is not augmented yet, take original image to apply the next augmentation
							filename = args.image_dir + filename
						else: # else take the image that is previously augmented (and add on new augmentations on it)
							filename = filename_new
						# assign image augmenatations
						new_ratio = image_augmentation(operation, filename, filename_new, len(operations[i_op]), reduce, args.dataset)

						# only change the aspect ratio if a valid value was return
						if new_ratio is not None:
							aspect_ratio = new_ratio

					# annotations augmentation
					xml_file = args.xml_dir + img_list[i_im][:-3] + "xml"
					tree = ET.parse(xml_file)
					root = tree.getroot()
					# getting all bounding boxes from the xml annotation file
					for member in root.findall('object'): # for each instance
						width = int(root.find('size')[0].text)
						height = int(root.find('size')[1].text)
						xmin = int(member[4][0].text)
						ymin = int(member[4][1].text)
						xmax = int(member[4][2].text)
						ymax = int(member[4][3].text)
						label = member[0].text
						# assign annotations augmentation
						width, height, xmin, ymin, xmax, ymax = annotation_augmentation(operations[i_op], width, height, xmin, ymin, xmax, ymax, aspect_ratio)
						# Special case: if random erase need to be applied (it is special as it needs bounding box information, but do not augment the annotations)
						if "Erase" in operations[i_op]:
							aug.random_erase(filename, filename_new, xmin, xmax, ymin, ymax)
							filename = filename_new # take the image that was augmented after the first time appling random erase
						# write into csv
						j = filename_new.rfind("/")
						image_name = filename_new[j+1:]
						writer.writerow([image_name, height, width, xmin, ymin, xmax, ymax, label])

# run code if it is run as main code (not imported in other scripts)
if __name__ == "__main__":
    main(args)
