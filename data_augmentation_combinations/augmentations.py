'''
This code includes all image and annotation augmentations functions that are used in augmentation_per_dataset.py
To read more about image augmentation, read: https://towardsdatascience.com/data-augmentation-in-yolov4-c16bd22b2617
Written by: Tong Siew Wen (January 2021)

How to use:
- this code does not has a main function, it is to be imported to other code so that these functions can be used
- can be imported and called as follow:
	...
	import augmentations as aug
	# lots of codes
	aug.rotate_image(input_parameters) # calling functions in this file
	...

What is required:
- all the modules that are imported in this file should be installed (pip3 install <module_name>)

List of functions:
- rotate_box -> 90, 180, 270 degree clockwise ONLY
- rotate_image -> 90, 180, 270 degree clockwise ONLY
- flip_box -> flip vertically or horizontally ONLY
- flip_image -> flip vertically or horizontally ONLY
- change_aspect_ratio_box
- change_aspect_ratio_image
- shear_box
- shear_image
- change_hue
- change_saturation
- change_brightness
- change_contrast
- add_noise -> salt and peper noise
- blur_image -> Gaussian blur
- random_erase
- grid_mask

To take note:
- xxx_box augments the bounding box coordinate, xxx_image augments the image
- all functions that augments the image, saves the image as filename_new
- all the functions do not change the original image array
- feel free to add other functions to this code (it will not affect the original code)
	- but remember that we dont want any code in this file to run when being imported to augmentation_per_dataset.py
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
import random
def rotate_box(w, h, minx, miny, maxx, maxy, rotation_degree):
	'''
	Function: Augment annoatations according to the rotation degree inputed
	...
	Input parameters:
	rotation_degree = 90 or 180 or 270 (only accepts these three degrees)
	w = width of image (integer only)
	h = height of image (integer only)
	minx = minimum x coordinate of bounding box (integer only)
	miny = minimum y coordinate of bounding box (integer only)
	maxx = maximum x coordinate of bounding box (integer only)
	maxy = maximum y coordinate of bounding box (integer only)
	...
	Return values:
	width = new width of image (integer only)
	height = new height of image (integer only)
	xmin = new minimum x coordinate of bounding box (integer only)
	ymin = new minimum y coordinate of bounding box (integer only)
	xmax = new maximum x coordinate of bounding box (integer only)
	ymax = new maximum y coordinate of bounding box (integer only)
	'''
	if rotation_degree == 90:
		width = h
		height = w
		xmin = h-maxy
		ymin = minx
		xmax = h-miny
		ymax = maxx
	elif rotation_degree == 180:
		width = w
		height = h
		xmin = w-maxx
		ymin = h-maxy
		xmax = w-minx
		ymax = h-miny
	elif rotation_degree == 270:
		width = h
		height = w
		xmin = miny
		ymin = w-maxx
		xmax = maxy
		ymax = w-minx
	else:
		raise ValueError("degree of rotation must be 90 180 or 270 only")
	return width, height, xmin, ymin, xmax, ymax

def rotate_image(filename, filename_new, rotation_degree):
	'''
	Function: Rotate image according to the rotation degree inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	rotation_degree = cv2.ROTATE_90_CLOCKWISE (0) or cv2.ROTATE_180 (1) or cv2.ROTATE_90_COUNTERCLOCKWISE (2) (accepts these three values only)
	'''
	degrees = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
	if rotation_degree not in degrees: # checking whether the input for rotation_degree is valid
		raise ValueError("degree of rotation must be 0, 1, 2 only")
	image = cv2.imread(filename)
	rotated = cv2.rotate(image, rotation_degree) # rotate image
	cv2.imwrite(filename_new, rotated)

def flip_box(w, h, minx, miny, maxx, maxy, flip_code):
	'''
	Function: Augment annoatations according to the flip direction inputed
	...
	Input parameters:
	w = width of image (integer only)
	h = height of image (integer only)
	minx = minimum x coordinate of bounding box (integer only)
	miny = minimum y coordinate of bounding box (integer only)
	maxx = maximum x coordinate of bounding box (integer only)
	maxy = maximum y coordinate of bounding box (integer only)
	flip_code = any integer in the range [0, +infinity] (vertical: 0, horizontal: > 0)
	...
	Return values:
	width = new width of image (integer only)
	height = new height of image (integer only)
	xmin = new minimum x coordinate of bounding box (integer only)
	ymin = new minimum y coordinate of bounding box (integer only)
	xmax = new maximum x coordinate of bounding box (integer only)
	ymax = new maximum y coordinate of bounding box (integer only)
	'''
	if flip_code == 0: # flip vertically
		xmin = minx
		ymin = h-maxy
		xmax = maxx
		ymax = h-miny
	elif flip_code > 0: # flip horizontally
		xmin = w-maxx
		ymin = miny
		xmax = w-minx
		ymax = maxy
	else:
		raise ValueError("this code do not handle flip vertical + horizontal. Use rotation 270 degree clockwise instead")
	# width and height are unchanged
	width = w
	height = h
	return width, height, xmin, ymin, xmax, ymax

def flip_image(filename, filename_new, flip_code):
	'''
	Function: Flip image according to the flip direction, and save the augmented image inputed
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	flip_code = any integer between range [0, +infinity] (vertical: 0, horizontal: > 0)
	'''
	image = cv2.imread(filename)
	flipped = cv2.flip(image, flip_code) # flip image (vertical: flip_code = 0, horizontal: flip_code > 0)
	cv2.imwrite(filename_new, flipped)

def change_aspect_ratio_box(width, height, xmin, xmax, ymin, ymax, aspect_ratio):
	'''
	Function: Augment annoatations according to the aspect ratio inputed
	...
	Input parameters:
	width = width of image (integer only)
	height = height of image (integer only)
	xmin = minimum x coordinate of bounding box (integer only)
	ymin = minimum y coordinate of bounding box (integer only)
	xmax = maximum x coordinate of bounding box (integer only)
	ymax = maximum y coordinate of bounding box (integer only)
	aspect_ratio = float that represents width:height (e.g aspect ratio = 1.5 -> width:height = 3:2)
	...
	Return values:
	width = new width of image (integer only)
	height = new height of image (integer only)
	xmin = new minimum x coordinate of bounding box (integer only)
	ymin = new minimum y coordinate of bounding box (integer only)
	xmax = new maximum x coordinate of bounding box (integer only)
	ymax = new maximum y coordinate of bounding box (integer only)
	'''
	w = width
	width = math.ceil(height*aspect_ratio) # change the width of image
	ratio = width/w*1.0 # find out how much has the width change
	# change the value of xmin and xmax (ymin and ymax are unchanged as height of image is unchanged)
	xmin = math.ceil(xmin*ratio)
	xmax = math.ceil(xmax*ratio)
	return width, height, xmin, ymin, xmax, ymax

def change_aspect_ratio_image(filename, filename_new, aspect_ratio):
	'''
	Function: Change aspect ratio of image according to the aspect ratio inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	aspect_ratio = float that represents width:height (e.g aspect ratio = 1.5 -> width:height = 3:2)
	'''
	image = cv2.imread(filename)
	height = image.shape[0]
	width = math.ceil(height*aspect_ratio)
	output = cv2.resize(image,(width,height)) # resize image to the aspect ratio inputed
	cv2.imwrite(filename_new, output)

# shear box and image need to add in two more directions
def shear_box(width, height, xmin, xmax, ymin, ymax, direction, shear_degree):
	'''
	Function: Augment annoatations according to the shear direction and degree inputed
	...
	Input parameters:
	width = width of image (integer only)
	height = height of image (integer only)
	xmin = minimum x coordinate of bounding box (integer only)
	ymin = minimum y coordinate of bounding box (integer only)
	xmax = maximum x coordinate of bounding box (integer only)
	ymax = maximum y coordinate of bounding box (integer only)
	direction = interger in range of [0,3] (directions are illustrated below)
	shear_degree = float in range of (0,1.57) radians (equivalant to 0 to 90 degrees)
	...
	Directions:
	original  ->    0      1      2     >= 3
	   __           __    __      /|     |\       .
	  |__|    ->   /_/    \_\    |/       \|
	...
	Return values:
	width = new width of image (integer only)
	height = new height of image (integer only)
	xmin = new minimum x coordinate of bounding box (integer only)
	ymin = new minimum y coordinate of bounding box (integer only)
	xmax = new maximum x coordinate of bounding box (integer only)
	ymax = new maximum y coordinate of bounding box (integer only)
	'''
	if direction < 2 : # if shear horzontally
		extra_width = math.ceil(height*math.sin(shear_degree))
		width = width+extra_width # change image width
		if (direction == 0):
			xmin = math.floor(xmin + ymin*math.sin(shear_degree))
			xmax = math.floor(xmax + ymax*math.sin(shear_degree))
		else:
			xmin = math.floor(xmin + (height-ymax)*math.sin(shear_degree))
			xmax = math.floor(xmax + (height-ymin)*math.sin(shear_degree))
	else: # if shear vertically
		extra_height= math.ceil(width*math.sin(shear_degree))
		height = height+extra_height # change image height
		if (direction == 2):
			ymin = math.floor(ymin + xmin*math.sin(shear_degree))
			ymax = math.floor(ymax + xmax*math.sin(shear_degree))
		else:
			ymin = math.floor(ymin + (width-xmax)*math.sin(shear_degree))
			ymax = math.floor(ymax + (width-xmin)*math.sin(shear_degree))
	return width, height, xmin, ymin, xmax, ymax

def shear_image(filename, filename_new, direction, shear_degree):
	'''
	Function: Augment image according to the shear direction and degree inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	direction = interger in range of [0,3] (directions are illustrated below)
	shear_degree = float in range of (0,1.57) radians (equivalant to 0 to 90 degrees)
	...
	Directions:
	original  ->    0      1      2     >= 3
	   __           __    __      /|     |\      .
	  |__|    ->   /_/    \_\    |/       \|

	'''
	image = cv2.imread(filename)
	width = image.shape[1]
	height = image.shape[0]
	if direction < 2: # if shear horizontally
		extra_width = math.ceil(height*math.sin(shear_degree))
		shape = (height, width+extra_width, image.shape[2])
		output = np.zeros(shape,np.uint8) # create empty array with wider width to allow shearing
		for i in range(height):
			shear = math.floor(i*math.sin(shear_degree)) if direction == 0 else math.floor((height-i)*math.sin(shear_degree))
			for j in range(width):
				output[i][j+shear] = image[i][j] # shear image
	else: # if shear vertically
		extra_height= math.ceil(width*math.sin(shear_degree))
		shape = (height+extra_height, width, image.shape[2])
		output = np.zeros(shape,np.uint8) # create empty array with wider height to allow shearing
		for i in range(height):
			for j in range(width):
				shear = math.floor(j*math.sin(shear_degree)) if direction == 2 else math.floor((width-j)*math.sin(shear_degree))
				output[i+shear][j] = image[i][j] # shear image
	cv2.imwrite(filename_new, output)

def change_hue(filename, filename_new, hout):
	'''
	Function: Shift the hue of the image according to the hue shift inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	hout = float in the range (0,1) (amount of hue shifted) note: hout = 0 or 1 will not change the hue of the image
	'''
	# vertorise the convertion between rgb and hsv
	rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
	hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
	image = Image.open(filename)
	img = image.convert('RGBA')
	# get r, g, b value of image
	arr = np.array(np.asarray(img).astype('float'))
	r, g, b, a = np.rollaxis(arr, axis=-1)
	# convert rgb values to hsv values
	h, s, v = rgb_to_hsv(r, g, b)
	# shift hue
	h = (h + hout) % 1
	# convert hsv values to rgb values
	r, g, b = hsv_to_rgb(h, s, v)
	# create augmented image with hue shifted
	arr = np.dstack((r, g, b, a))
	new_img = Image.fromarray(arr.astype('uint8'), 'RGBA')
	new_img = new_img.convert('RGB')
	new_img.save(filename_new, quality=95)

def change_saturation(filename, filename_new, saturation_shift):
	'''
	Function: Change saturation of the image according to the saturation shift inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	saturation_shift = float in the range [0, +infinity] (value < 1 decrease the saturation, value > 1 increase the saturation)
	'''
	img = Image.open(filename)
	# get color of original image
	converter = ImageEnhance.Color(img)
	# change saturation
	new_img = converter.enhance(saturation_shift)
	new_img = new_img.convert('RGB')
	new_img.save(filename_new, quality=95)

def change_brightness(filename, filename_new, brightness_shift):
	'''
	Function: Change brightness of the image according to the brightness shift inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	brightness_shift = float in the range [0, +infinity] (value < 1 decrease the brightness, value > 1 increase the brightness)
	'''
	# print("brightness shift = " + str(brightness_shift))
	img = Image.open(filename)
	# get brightness of original image
	converter = ImageEnhance.Brightness(img)
	# change brightness
	new_img = converter.enhance(brightness_shift)
	new_img = new_img.convert('RGB')
	new_img.save(filename_new, quality=95)

def change_contrast(filename, filename_new, contrast_shift):
	'''
	Function: Change contrast of the image according to the contrast shift inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	contrast_shift = float in the range [0, +infinity] (value < 1 decrease the contrast, value > 1 increase the contrast)
	'''
	# print("contrast shift = " + str(contrast_shift))
	img = Image.open(filename)
	# get contrast of original image
	converter = ImageEnhance.Contrast(img)
	# change contrast
	new_img = converter.enhance(contrast_shift)
	new_img = new_img.convert('RGB')
	new_img.save(filename_new, quality=95)

def add_noise(filename, filename_new, prob):
	'''
	Function: Add salt and pepper noise to image according to the probability of the noise inputed, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	prob = Probability of the noise -> float in the range [0, 1] (the greater the value, the greater the amount of noise added)
	'''
	image = cv2.imread(filename)
	thres = 1 - prob
	# create new image (empty array)
	output = np.zeros(image.shape,np.uint8)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			rdn = random.random()
			# change image pixel to black with probablity of prob
			if rdn < prob:
			    output[i][j] = 0
			# change image pixel to white with probablity of prob
			elif rdn > thres:
			    output[i][j] = 255
			else:
				output[i][j] = image[i][j]
	cv2.imwrite(filename_new, output)

def blur_image(filename, filename_new, blur_radius):
	'''
	Function: Blur the image according to the blur radius inputed using Gaussian blur, and save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	blur_radius = any positive integer (the greater the value, the blurer the image becomes)
	'''
	OriImage = Image.open(filename)
	# apply Guassian Blur on image
	blurImage = OriImage.filter(ImageFilter.GaussianBlur(blur_radius))
	blurImage = blurImage.convert('RGB')
	blurImage.save(filename_new)

def random_erase(filename, filename_new, xmin, xmax, ymin, ymax):
	'''
	Function: replaces regions within the bounding box with random values, with varying proportion and aspect ratio, then save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	xmin = minimum x coordinate of bounding box (integer only)
	ymin = minimum y coordinate of bounding box (integer only)
	xmax = maximum x coordinate of bounding box (integer only)
	ymax = maximum y coordinate of bounding box (integer only)
	'''
	# computing width and height of bounding box
	w_box = xmax - xmin
	h_box = ymax - ymin
	# randomly choose width and height of the occluding block
	w = random.randint(math.floor(w_box*0.3), math.ceil(w_box*0.5))
	h = random.randint(math.floor(h_box*0.3), math.ceil(h_box*0.5))
	# randomly choose position of the occluding block
	start_x = random.randint(xmin, xmax-w)
	start_y = random.randint(ymin, ymax-h)

	image = cv2.imread(filename)
	output = np.zeros(image.shape,np.uint8)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if (j > start_x and j < start_x + w) and (i > start_y and i < start_y + h):
				# if the pixel lies in the position of the occluding block, change the image pixel to a random value
				output[i][j] = random.randint(0,255)
			else:
				output[i][j] = image[i][j]
	cv2.imwrite(filename_new, output)

def grid_mask(filename, filename_new):
	'''
	Function: Regions of the image are hidden in a grid like fashion, with a randomly chosen grid size, then save the augmented image
	...
	Input parameters:
	filename = path to image to be augmented (relative to current working directory) -> e.g. image_dir/image_name.jpg
	filename_new = path to save the augmented image (relative to current working directory) -> e.g. save_dir/new_name.jpg
	'''
	image = cv2.imread(filename)
	# randomly choose the size of the grid ofr the grid mask
	size = min(image.shape[0], image.shape[1])
	w = random.randint(math.floor(size/100), math.ceil(size/50))
	# randomly choose position of the top left grid on the image
	start_x = random.randint(0, w)
	start_y = random.randint(0, w)

	output = np.zeros(image.shape,np.uint8)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if (j > start_x) and (i > start_y):
				# starting from the position of the top left grid
				if (j - start_x) % (2*w) < w and (i - start_y) % (2*w) < w:
					# if the pixel lies in the position of the grid, change the image pixel to black
					output[i][j] = 0
				else:
					output[i][j] = image[i][j]
			else:
				output[i][j] = image[i][j]
	cv2.imwrite(filename_new, output)
