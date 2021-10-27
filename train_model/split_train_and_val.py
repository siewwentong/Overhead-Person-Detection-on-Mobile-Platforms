'''
This script splits training and validation image and annotations from csv_input according to validation_file
into final_train/train_csv_output and final_train/images, as well as final_val/val_csv_output and final_val/images.
I have edited the code to have the option to include synthetic images and augmented images, and rename validation images with random names.
Edited by Tong Siew Wen (December 2020)
...
How to use this code:
- python3 split_train_and_val.py --validation_file <name of textfile> --csv_input <name of csv file> (add other parameters if needed)
- this script will create final_train and final_val folder
...
What is required:
- a list (textfile) that has all the image names to be the validation set (input for --validation_file)
- a csv file that contains all annotations (training and validation) (input for --csv_input)
- folder for augmented images (optional)
- folder for synthetic images (only needed if --synthetic == 1 )
...
To take note:
- the folder to get synthetic images and augmented images are "synthetic_images/" and "augmented_images/",
  please change them in the script if you kept them in different folders
- synthetic images will replace the original images if --synthetic == 1
- IMPORTANT: everytime you run this script, train_csv_output and val_csv_output will be overwrite the csv that has the same name.
			 It will also overwrite images of the same name. If you want to keep the previous data, please rename them.
'''

import sys
import csv
import cv2
import os
from shutil import copyfile

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--validation_file")
parser.add_argument("--csv_input")
parser.add_argument("--train_csv_output", default="train.csv")
parser.add_argument("--train_csv_output", default="val.csv")
parser.add_argument("--synthetic", help="use normal or synthetic images (True: 1/ False: 0)")
args = parser.parse_args()

for_val = []

# NOTE: change name of folders if you are using other folders to keep these images
src = "synthetic_images/" if (int(args.synthetic) == 1) else "images/" # priotised source
src_aug = "augmented_images/" # source for augmented images
dst_val = "final_val/images/" # destination for validation images
dst_train = "final_train/images/" # destination for train images

with open(args.validation_file, 'r') as file:
	lines = file.readlines()
	for line in lines:
		image_name = line[:-1]
		for_val.append(image_name) # get a list of validation images

# create folders if they does not exixt
if not os.path.exists("final_train/"):
	os.makedirs("final_train/")
	os.makedirs("final_train/images/")
if not os.path.exists("final_val/"):
	os.makedirs("final_val/")
	os.makedirs("final_val/images/")
if not os.path.exists("final_train/images/"):
	os.makedirs("final_train/images/")
if not os.path.exists("final_val/images/"):
	os.makedirs("final_val/images/")

with open(args.csv_input, 'r') as file:
	with open('final_val/' + args.val_csv_output, 'w') as val_file:
		with open('final_train/' + args.train_csv_output, 'w') as train_file:
			reader = csv.reader(file)
			writer_val = csv.writer(val_file)
			writer_train = csv.writer(train_file)
			for row in reader:
				filename = row[0]
				# for validation
				if (filename in for_val):
					# rename validation images with random values (their index in validation_file)
					count = str(for_val.index(filename))
					new_name = count.zfill(4) + ".jpg"
					row[0] = new_name
					# copy image from their source to final_val/images/
					if not (os.path.isfile(dst_val+new_name)):
						print("validation: " + filename)
						if (os.path.isfile(src+filename)):
							copyfile(src+filename, dst_val+new_name)
						elif (os.path.isfile(src_aug+filename)):
							copyfile(src_aug+filename, dst_val+new_name)
						elif (os.path.isfile("images/"+filename)):
							copyfile("images/"+filename, dst_val+new_name)
						else:
							raise ValueError("file not found: " + filename)
					# write annotations into val_csv_output
					writer_val.writerow(row)]
				# for training
				else:
					# copy image from their source to final_val/images/
					if not (os.path.isfile(dst_train+filename)):
						print("train: " + filename)
						if (os.path.isfile(src+filename)):
							copyfile(src+filename, dst_train+filename)
						elif (os.path.isfile(src_aug+filename)):
							copyfile(src_aug+filename, dst_train+filename)
						elif (os.path.isfile("images/"+filename)):
							copyfile("images/"+filename, dst_train+filename)
						else:
							raise ValueError("file not found: " + filename)
					# write annotations into val_csv_output
					writer_train.writerow(row)
