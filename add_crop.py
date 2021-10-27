from PIL import Image
import glob
import xml.etree.ElementTree as ET
import random
import os
import numpy as np
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ori_im_dir")
parser.add_argument("--ori_xml_dir")
parser.add_argument("--crop_im_dir")
parser.add_argument("--output_dir")
args = parser.parse_args()

manual_xy = []

def manual_choose_xy(event, x, y, flags, param):
	global manual_xy
	if event == cv2.EVENT_LBUTTONDOWN:
		manual_xy = [x, y]
		print("choosen: ", manual_xy)

images = glob.glob(str(args.ori_im_dir) + "*.jpg")
crops = glob.glob(str(args.crop_im_dir) + "*.png")
print(len(crops))

if not os.path.exists(str(args.output_dir)):
    os.mkdir(str(args.output_dir))

count = 0
for idx, image in enumerate(images):
	im_name = image[image.rfind("/")+1:]
	print(im_name)
	xml = str(args.ori_xml_dir) + im_name[:-4] + ".xml"
	# get info from xml file
	tree = ET.parse(xml)
	root = tree.getroot()
	minxs = []
	minys = []
	maxxs = []
	maxys = []
	im_height = int(root.find('size')[1].text)
	im_width = int(root.find('size')[0].text)
	for member in root.findall('object'): # for each instance
		minxs.append(int(member[4][0].text))
		minys.append(int(member[4][1].text))
		maxxs.append(int(member[4][2].text))
		maxys.append(int(member[4][3].text))

	# pick a random coordinate to place the human instances
	x = random.randint(0,im_width-265)
	y = random.randint(0,im_height-265)

	flag=False
	j = 0
	# print(minxs)
	# print(minys)
	# print(maxxs)
	# print(maxys)

	im_size = max(maxxs[0] - minxs[0] + 50, maxys[0] - minys[0] + 50)
	min_size = im_size*3/4
	combined = Image.open(image)

	while(True):
		i = 0
		for minx, miny, maxx, maxy in zip(minxs, minys, maxxs, maxys):
			# if (x,y) in the box
			if (x+im_size > minx and x < maxx) and (y + im_size > miny and y < maxy) or x+im_size > im_width or y+im_size > im_height :
				# repick (x,y) and start all over
				x = random.randint(0,im_width-265)
				y = random.randint(0,im_height-265)
				j+=1
				break
			else:
				i += 1
		if i == len(minxs):
			cropped = Image.open(crops[idx])
			cropped = cropped.resize((im_size,im_size))
			combined.paste(cropped, (x, y))
			break
		if j == 500:
			im_size -= 10
			print("reducing size...")
			if im_size < min_size:
				im_size = int(min_size)
				print("manually choosing a point")
				im_cv2 = cv2.imread(image)
				cv2.imshow("image", im_cv2)
				cv2.setMouseCallback("image", manual_choose_xy)
				cv2.waitKey(0)
				x, y = manual_xy
				test_im = Image.open(image)
				test_crop = Image.open(crops[idx])
				test_crop = test_crop.resize((im_size,im_size))
				test_im.paste(test_crop, (x, y))
				test_im_cv2 = np.array(test_im)
				test_im_cv2 = test_im_cv2[:, :, ::-1].copy()
				cv2.imshow("synthetic image", test_im_cv2)
				cv2.waitKey(0)
				# print(minxs)
				# print(minys)
				# print(maxxs)
				# print(maxys)
				# for minx, miny, maxx, maxy in zip(minxs, minys, maxxs, maxys):
				# 	# if (x,y) in the box
				# 	print("Is " + str(x+im_size) + " > " + str(minx) + "and " + str(x) + " < " + str(maxx,) + "and" +  str(y+im_size) + " > " + str(miny) + " and " + str(y) + " < " + str(maxy) + "?" )
				# 	if (x+im_size > minx and x < maxx) and (y + im_size > miny and y < maxy):
				# 		print("Yes")
				# 	else:
				# 		print("No")
				# exit()
				# print("skipped: " + im_name)
				# count+=1
				# break
			j = 0

	combined.save(str(args.output_dir) + im_name, quality=95)

	# # to view the synthetic image
	# ori_im = cv2.imread(image)
	# combined_cv2 = np.array(combined)
	# combined_cv2 = combined_cv2[:, :, ::-1].copy()
	# cv2.imshow("original image", cv2.resize(ori_im, (640,480)))
	# cv2.imshow("synthetic image", cv2.resize(combined_cv2, (640,480)))
	# cv2.waitKey(0)
print(count)
