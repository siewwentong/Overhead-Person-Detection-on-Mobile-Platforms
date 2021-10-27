'''
This script creates a list (textfile) of settings
Written by Tong Siew Wen
...
How to run the code:
- python3 create_setting_list.py
...
What is required:
- split your images into different folders according to the setting of the image (one folder per setting)
	- for these folders, it should consist all the images that have the same background so that a good background model can be created for each setting
		- see documentation in google doc: https://docs.google.com/document/d/1jrgZaX9pGhLj_1_e4d1v4v2Mg6JO6Z4RnleJnKUcMMg/edit?usp=sharing
	- note: collecting image on different day might result a different background too, even though it is taken at the same location
	- note: brightness, height etc. should also be taken into consideration (same location doesn't mean same background!)
'''
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_list", default = "all_datasets.txt")
parser.add_argument("--dataset_dir", default = "datasets/")
args = parser.parse_args()

# getting paths to all the settings folders
settings = glob.glob( args.dataset_dir + "*/setting*/")
settings.sort()
with open(args.dataset_list,'w') as txtfile:
	for setting in settings:
		# write each path to the text file
		txtfile.write(setting[:-1] + "\n")
