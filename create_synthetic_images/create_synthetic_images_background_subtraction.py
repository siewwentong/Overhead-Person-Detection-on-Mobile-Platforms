'''
This code creates synthetc images, and saves it using the same name as the original image in output_dir folder
Written by: Tong Siew Wen (January 2021)

How to run the code:
- python3 create_synthetic_images_background_subtraction.py (add in paprameters if needed, explained in "What is required below")
- The code will first create one object per setting (each object has its name, list of images and its background model)
- Then each object's background mdoel will be trained (you will see the line "training model blablabla ...")
- After training, each image will be attempted to be created into a synthetic image
- You will see three pop-up windows for each image (original image, synthetic image and synthetic image with bounding box drawn)
- Based on your judgement, you will then need to decide whether you want to save it
	- pressing "s" key will save the image
	- pressing "q" key will end the program (quit)
	- pressing other key will discard this image

What is required:
- a list (textfile) of background images to be used (input for --background_list)
- a folder that consist of all background images to be used (input for --background_dir)
- a folder that consists of all the images (input for image_dir)
- a folder that consists of all the xml annotation files for the images in image_dir (xml file need to be of the same name as image) (input for --xml_dir)
- a list (textfile) of all the datasets/settings that are used (each line must be in the form: bla/dataset_name/setting_name, see "To take note" below) (input for --dataset_list)

To take note:
- This code will not remember which is the last image you are working on (meaning each time you run the code, it'll start over from the very first image)
- Once you decided to discard an image, you can't go back (unless you re-run the code)
- when writing inputs for --*_dir, it must end with "/" (e.g. --image_dir images/)
- for dataset_list:
	- the path is relative to the current working directory
	- you need to have folders that correspond to each of the line in the textfile (e.g. ../datasets/EC/setting1 really exixts)
		- for these folders, it should consist all the images that have the same background so that a good background model can be created for each setting
		- note: collecting image on different day might result a different background too, even though it is taken at the same location
		- note: brightness, height etc. should also be taken into consideration (same location doesn't mean same background!)
- For debuging purpose, you can uncomment lines with "print(...)"" or "cv2.imshow(...)"
- you can change the default value for the flags so it is easier to call the code (don't need to input each parameter)
- it is recommended that you make a copy of the original code before you make any changes to the code
'''


import csv
from PIL import Image, ImageDraw, ImageFilter
from shutil import copyfile
import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--background_list", default = 'backgrounds.txt')
parser.add_argument("--background_dir", default = "backgrounds/")
parser.add_argument("--image_dir", default = "images/")
parser.add_argument("--xml_dir", default = "images/")
parser.add_argument("--output_dir", default = "synthetic_images/")
parser.add_argument("--dataset_list", default = "all_datasets.txt")
args = parser.parse_args()

# train background model -> train one model for each background (done)
# iterate through each dataset list -> array (done)
# split synthetic images according to their datasets (done)
# make create syntheic images as a fucntion (mask, images name, dataset name) (done)
# if image from csv in one of the datset, apply on model for that dataset to get the mask, create synthetic_image

def create_synthetic_image(image_name, index, mask_name):
	'''
	FUnction: creates a synthetic image and let you decide whether to save the image or not (See "How to run the code" above)
	...
	Input parameter:
	- image_name = only the name of the image, e.e. xxx.jpg (NOT path to image, e.g. xxx/xxx.jpg)
	- index = chooses which background image to use (refers to the index in the background list)
	- mask_name = name of the mask that was created in the main code. Default: "mask.jpg"
	'''

	with open(args.background_list, 'r') as txt:
		backgrounds = txt.readlines()

		fore = Image.open(args.image_dir + image_name)
		width, height = fore.size

		foreground_mask = Image.open(mask_name).resize((width,height)).convert('L')
		# mask = foreground_mask.filter(ImageFilter.GaussianBlur(4))

		# if number of images to be converted is more than the number of background
		while (index > len(backgrounds) - 1):
			index = index-len(backgrounds)+1
		background = backgrounds[index][:-1]

		# check whether a synthetic image has been created for this image
		if (os.path.isfile(args.output_dir+ image_name)):
			# if yes, we are adding new instances to the picture
			back = Image.open(args.output_dir + image_name)
		else: #if no, we are creating a new synthetic image
			back = Image.open(args.background_dir + background)

		combined = back.copy()
		combined = combined.resize((width,height))
		# get info from xml file
		xml_file = args.xml_dir + image_name[:-3] + "xml"
		tree = ET.parse(xml_file)
		root = tree.getroot()
		startPoints = []
		endPoints = []
		for member in root.findall('object'): # for each instance
			minx = int(member[4][0].text)
			miny = int(member[4][1].text)
			maxx = int(member[4][2].text)
			maxy = int(member[4][3].text)

			# crop human out
			cropped = fore.crop((minx, miny, maxx, maxy))
			mask = foreground_mask.crop((minx, miny, maxx, maxy))
			# paste human onto background image with applied mask
			combined.paste(cropped, (minx, miny), mask)

			startPoints.append((minx,miny))
			endPoints.append((maxx,maxy))

		combined = combined.convert('RGB')

		# to view the synthetic image
		combined_cv2 = np.array(combined)
		combined_cv2 = combined_cv2[:, :, ::-1].copy()
		cv2.imshow("synthetic image", combined_cv2)
		for index, startPoint in enumerate(startPoints):
			# draw bounding boxes
			cv2.rectangle(combined_cv2,startPoint,endPoints[index],(255,0,0),3)
		cv2.imshow("synthetic image with bounding box", combined_cv2)

		res = cv2.waitKey(0)
		#print(res)
		if (res == ord('s')): # if key 's' is pressed, save
			combined.save(args.output_dir+ filename, quality=95)
			print("created: " + args.output_dir + " " + filename + " with background " + background)
			return True
		elif (res == ord('q')): # if key 'q' is pressed, discard and quit
			exit()
		else: # if other key is pressed, discard
			return False

def create_dataset_array(dataset):
	'''
	Function: returns a list of images that is in the folder that correspond to that setting
	...
	Input parameter:
	dataset = path to the folder that consist all the images that have the same background (relative to the current working directory)
	'''
	images = glob.glob(dataset + "/*.jpg")
	images.sort()
	img_list = []
	for image in images:
		i3 = image.rfind("/")
		img_list.append(image[i3+1:])
	return img_list

def train_background_model(dataset, background_model, history):
	'''
	Function: trains the background model based on the images that was fed in
	...
	Input parameter:
	- dataset = path to the consist all the images that have the same background (relative to the current working directory)
	- background_model = MOG2 background model that has already been created
	- history = paremater that was used to create this background model
	'''
	i = dataset.rfind("/")
	setting_name = dataset[i+1:]
	i2 = dataset[:i].rfind("/")
	dataset_name = dataset[i2+1:i]
	images = glob.glob(dataset + "/*.jpg")
	images.sort()
	print("training model " + dataset_name + " " + setting_name + "...")
	# for index, image in enumerate(images):
	for index in range(history):
		#train
		image = cv2.imread(images[index])
		image = cv2.resize(image, (640, 480))
		mask = background_model.apply(image)

class Dataset:
	'''
	Object for each setting
	...
	conists of setting name, list of images for this setting, and its background model
	history is chosen based on the number of images for the setting
	(if number of images is more than 500, then history is 500, else history is the total number of images for that setting)
	'''
	def __init__(self, name, img_list):
		self.name = name
		self.img_list = img_list
		self.history = min(500, len(img_list))
		self.background_model = cv2.createBackgroundSubtractorMOG2(self.history, 16, True)

def fillhole(input_image):
	'''
	Function: returns an image that fills up holes in the input image that are fully enclosed
	...
	Input parameter:
	input image = a numpy array (openCv) binary image
	...
	Note:
	Only holes surrounded in the connected regions will be filled.
	Original image is not altered.
	Source: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

	'''
	# create an empty array with the same shape as original image, and pad zeros at the surrounding
	h, w = input_image.shape[:2]
	im_flood_fill = np.zeros((h + 2, w + 2), np.uint8)
	im_flood_fill[1:-1,1:-1] = input_image.copy()
	# apply floodfill
	mask = np.zeros((h + 4, w + 4), np.uint8)
	im_flood_fill = im_flood_fill.astype("uint8")
	cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
	# cv2.imshow("after floodfill", im_flood_fill)
	# cv2.waitKey(0)
	# invert image after floodFill, then or it with the original image
	im_flood_fill_inv = cv2.bitwise_not(im_flood_fill[1:-1,1:-1])
	img_out = input_image | im_flood_fill_inv
	return img_out

'''
Main body of the code:
- The code will first create one object per setting (each object has its name, list of images and its background model)
- Then each object's background mdoel will be trained (you will see the line "training model blablabla ...")
- After training, each image will be attempted to be created into a synthetic image (and you have to decide whether to save the image or not)
'''
def main():
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	index = 0
	with open(args.dataset_list, 'r') as txt:
		dataset_lists = txt.readlines()
		dataset_lists.sort()
		# getting the datasets name
		datasets = []
		for dataset_list in dataset_lists:
			i = dataset_list.rfind("/")
			setting_name = dataset_list[i+1:-1]
			i2 = dataset_list[:i].rfind("/")
			dataset_name = dataset_list[i2+1:i]
			name = dataset_name + " " + setting_name
			# create a new Dataset object
			new_dataset = Dataset(name,create_dataset_array(dataset_list[:-1]))
			# train the background model
			train_background_model(dataset_list[:-1], new_dataset.background_model, new_dataset.history)
			datasets.append(new_dataset)
			# print(new_dataset.name)
			# print(len(new_dataset.img_list))

		# try to do synthetic images for all images
		counter = 0
		for dataset in datasets:
			for filename in dataset.img_list:
				image = cv2.imread(args.image_dir+filename)
				h, w = image.shape[:2]
				image = cv2.resize(image, (640, 480))
				cv2.imshow("image", image)
				cv2.waitKey(1)
				# apply background model to get foreground mask
				foreground_mask = dataset.background_model.apply(image)
				# cv2.imshow("mask", foreground_mask)
				# cv2.waitKey(1)

				# apply thresholding to get binary image (if > 200, pixel = 1, else 0)
				ret, foreground_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
				# cv2.imshow("after applying threshold", foreground_mask)
				# cv2.waitKey(1)

				'''
				to understand closing and opening, see:
				https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
				'''
				# applying closing to enclose holes that are not closed
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
				new_im = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
				# applying opening to remove noises
				new_im = cv2.morphologyEx(new_im, cv2.MORPH_OPEN, kernel)
				# applying closing to enclose holes that are not closed
				new_im = cv2.morphologyEx(new_im, cv2.MORPH_CLOSE, kernel)

				# cv2.imshow("mask after closing and opening", new_im)
				# cv2.waitKey(1)

				# fill holes that are onclosed
				new_im = fillhole(new_im)

				# cv2.imshow("mask ", new_im)
				# cv2.waitKey(1)

				cv2.imwrite("mask.jpg",new_im)
				# create synthetic images for each image
				saved = create_synthetic_image(filename, counter, "mask.jpg")
				if saved:
					# if saved, move on to use the next background image
					counter +=1

# run code if it is run as main code (not imported in other scripts)
if __name__ == "__main__":
    main()
