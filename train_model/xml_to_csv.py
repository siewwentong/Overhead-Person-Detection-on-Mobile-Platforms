'''
This code gathers all xml annotations from xml_dir into one single csv file (csv_output)
I edited the code to have an option to NOT convert images in "--list"
Edited by: Tong Siew Wen (January 2021)
...
How to use this script:
- python3 xml_to_csv.py --exlcude < 0 or 1 > --list <textfile name> (optional) (add other parameters if needed)
...
What is required:
- a folder that consist all the xml files (input to --xml_dir) [defult: images/]
- a list (textfile) of all the images to not include (only needed if --exclude == 1)
...
To take note:
- all image name is change to end with .jpg (the object detection pipeline only accepts jpg images)
  So please convert all your images to jpg image
- This script assumes name of xml file is same as image name
'''

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--xml_dir", help="path to xml files", default="images/")
parser.add_argument("--csv_output", help="path to output csv", default="tmp.csv")
parser.add_argument("--exclude", help="exclude images in list (True: 1, False: 0)")
parser.add_argument("--list", help="image list to not include")
args = parser.parse_args()
def xml_to_csv(do_not_include):
	'''
	Function: Converts xml annotations and writes into csv file
	...
	Input parameter:
	- do_not_include = a python list of all the images to not be included when converting
	...
	Return:
	xml_df = Data frame of all annotations
			 (a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns)
	'''
    xml_list = []
	# getting all xml files from xml_dir
    xml_files = glob.glob(args.xml_dir + '/*.xml')
    for xml_file in xml_files:
		# convert xml annotations not in the "to do include list"
        if xml_file not in do_not_include:
	        print(xml_file)
	        tree = ET.parse(xml_file)
	        root = tree.getroot()
			# get informations from xml file
	        for member in root.findall('object'):
	            filename = root.find('filename').text
	            i = filename.rfind(".")
				# filename only accpets jpg files (please convert images to jpg yourselves)
	            filename = filename[:i] + ".jpg"
	            value = (filename,
	                     int(root.find('size')[1].text),
	                     int(root.find('size')[0].text),
	                     int(member[4][0].text),
	                     int(member[4][1].text),
	                     int(member[4][2].text),
	                     int(member[4][3].text),
						 member[0].text
	                     )
	            xml_list.append(value)
        else:
	        print("exclude: " + xml_file)

    column_name = ['filename', 'height', 'width', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
	# convert xml annotations to Data frame
    xml_df = pd.DataFrame(xml_list)
    return xml_df

def exclude_list(txt_file, xml_dir):
	'''
	Function: Create a python list of images to not include according to txt_file
	...
	Input parameters:
	- txt_file = path to textfile (relative to curernt working directory)
	- xml_dir = path to folder that consists of all xml files
	...
	Return:
	- to_not_include = a python list of all the images to not be included when converting
	'''
	to_not_include = []
	with open (txt_file, 'r') as txt:
		list = txt.readlines()
		for item in list:
			to_not_include.append(str(xml_dir) + item[:-4] + "xml")
	return to_not_include

def main():
	aug = int(args.include_aug)
	do_not_include = exclude_list(args.list, args.xml_dir) if aug == 1 else []
	# convert to csv
	xml_df = xml_to_csv(do_not_include)
	xml_df.to_csv(args.csv_output, index=None)
	print('Successfully converted xml to csv.')

# run code if it is run as main code (not imported in other scripts)
if __name__ == "__main__":
    main(args)
