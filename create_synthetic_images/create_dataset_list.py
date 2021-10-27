'''
This script creates a list (textfile) per dataset of all the images in that dataset
Written by Tong Siew Wen
...
How to run the code:
- python3 create_dataset_list.py
...
What is required:
- split your images into different folders according to the dataset of the image (one folder per dataset)
	- see documentation in google doc: https://docs.google.com/document/d/1jrgZaX9pGhLj_1_e4d1v4v2Mg6JO6Z4RnleJnKUcMMg/edit?usp=sharing
'''
import glob
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default = "datasets/")
parser.add_argument("--dataset_list_dir", default = "dataset_list/")
args = parser.parse_args()

if not os.path.exists(args.dataset_list_dir):
	os.makedirs(args.dataset_list_dir)

# getting paths to all the dataset folders
datasets = glob.glob(args.dataset_dir + "*/")
datasets.sort()
for dataset in datasets:
	# extracting dataset name from path
	i = dataset.rfind("/")
	i2 = dataset[:i].rfind("/")
	dataset_name = dataset[i2+1:i]
	# create one text file per dataset (text file is named after the dataset name)
	txtfile_name = args.dataset_list_dir + dataset_name + ".txt"
	with open(txtfile_name,'w') as txtfile:
		# getting paths to all the images in each dataset
		images = glob.glob(dataset + "*.jpg")
		images2 = glob.glob(dataset + "*/*.jpg")
		all_images = []
		for image in images:
			i = image.rfind("/")
			image_name = image[i+1:]
			# ensuring image name is not repeated
			if image_name not in all_images:
				# write the image name into the text file
				all_images.append(image_name)
				txtfile.write(image_name + "\n")
		for image in images2:
			i = image.rfind("/")
			image_name = image[i+1:]
			# ensuring image name is not repeated
			if image_name not in all_images:
				# write the image name into the text file
				all_images.append(image_name)
				txtfile.write(image_name + "\n")
