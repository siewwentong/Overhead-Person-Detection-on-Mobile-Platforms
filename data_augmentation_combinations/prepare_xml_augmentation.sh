#!/bin/bash

# Written by: Tong Siew Wen (January 2021)
#
# How to use this code:
# - bash prepare_xml_augmentation.sh <data augmentation (True:1/False:0)>
# - This code will only convert annotations where their image are not augmented (i.e. synthetic images and original images that are kept)
#
# What is required:
# - flag to indicate whether to include images with data augmentation or not (True:1/False:0) when running this script
# - xml directory is default to "images/"
# 		- you can change this by adding "--xml_dir folder_name/" at the end of the line that calls xml_to_csv.py
# - a list of images that are augmented, to_augment.txt
#
# To take note:
# - Make sure all your images are in .jpg extention (please convert to jpeg if they are not)
# - This code assumes you have synthetic images in the folder "synthetic_images/"
# 		- if you dont have synthetic images:
# 				- change the following lines:
# 							...
# 							ls synthetic_images/ | grep ".jpg" | sort | uniq > tmp.txt
# 							cat to_augment.txt >> tmp.txt
# 							cat tmp.txt | sort | uniq -u > tmp2.txt
# 							...
# 					to
# 							cat to_augment.txt > tmp2.txt

# this code requires one input, hence if input parameter is not one, print error message and exit program
if [ "$#" -ne 1 ]; then
    echo "USAGE: bash prepare_xml_augmentation.sh <include augmentation (True:1/False:0)>"
		exit 1
fi

ls synthetic_images/ | grep ".jpg" | sort | uniq > tmp.txt
cat to_augment.txt >> tmp.txt
cat tmp.txt | sort | uniq -u > tmp2.txt

# converts annotations in xml files to single csv file
python xml_to_csv.py --include_aug $1 --list tmp2.txt
# if error occurs, quit program
if [ "$?" -ne 0 ]; then
		exit $?
fi

if [ "$1" == 1 ]
then # if include augmentation, then use output2.csv
	cat tmp.csv | tail -n +2 > tmp2.csv
	cat out_augmented.csv >> tmp2.csv
	cat tmp2.csv | sort -R > output2.csv
else # if use original annotations, then use output.csv
	cat tmp.csv | tail -n +2 | sort -R > output.csv
fi

# remove temporary files
rm tmp.txt
# if error occurs, quit program
if [ "$?" -ne 0 ]; then
		exit $?
fi

rm tmp2.txt
# if error occurs, quit program
if [ "$?" -ne 0 ]; then
		exit $?
fi

rm tmp.csv
# if error occurs, quit program
if [ "$?" -ne 0 ]; then
		exit $?
fi

rm tmp2.csv
# if error occurs, quit program
if [ "$?" -ne 0 ]; then
		exit $?
fi
