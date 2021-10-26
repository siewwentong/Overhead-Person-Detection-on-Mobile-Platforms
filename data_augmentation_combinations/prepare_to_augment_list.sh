#!/bin/bash
'''
Written by: Tong Siew Wen (January 2021)

How to use this code:
- bash prepare_to_augment_list.sh
- this code will prepare the list of images to be augmented in a textfile (to_augment.txt). It takes all images EXCEPT:
		- 20 original images per dataset
				- ensures that these images are not converted to synthetic images
				- this is because synthetic images are of the same name as the original image
				- in other words, it replaces the original images, hence these images are not "original" anymore
		- EDOB (as it is for validation)

What is required:
- a folder that consist of lists (textfiles) of images for each dataset
		- the folder I am using is "dataset_lists"
- Lists of images for each dataset
		- example: TVPR.txt -> consists of all the images in TVPR dataset
		- to know more about each dataset (person detection), please look into the documentation (google doc) I wrote (ELID Eye Training)
- a folder that consists of all synthetic images
		- the folder I am using is "synthetic_images/"
		- comment out the line "ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt" if you dont have synthetic images

To take note:
- This code is written speciffically for the datasets I am using, and can be used as reference to build your list for to_augment.txt
- This code keeps all the original images that are kept in a textfile, to_keep.txt
- To add a new dataset you need to copy-paste the block below, and replace "xxx" to the dataset name
			...
			cat dataset_list/xxx.txt | sort | uniq > tmp.txt
			ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt
			cat tmp.txt | sort | uniq -d > tmp2.txt
			cat dataset_list/xxx.txt >> tmp2.txt
			cat tmp2.txt | sort | uniq -u | sort -R | head -n 20 > to_keep.txt
			...
- To remove a dataset, you need to remove that whole block (see above) for that particular dataset
- It is recommended that you make a copy of the original code before you make any changes to the code
'''

# prepare the list of original images to keep in to_keep.txt
# keeps 20 original images for TVPR dataset (exclude those that are converted to synthetic images)
cat dataset_list/TVPR.txt | sort | uniq > tmp.txt
ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt
cat tmp.txt | sort | uniq -d > tmp2.txt
cat dataset_list/TVPR.txt >> tmp2.txt
cat tmp2.txt | sort | uniq -u | sort -R | head -n 20 > to_keep.txt
# keeps 20 original images for TVPR dataset (exclude those that are converted to synthetic images)
cat dataset_list/EC.txt | sort | uniq > tmp.txt
ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt
cat tmp.txt | sort | uniq -d > tmp2.txt
cat dataset_list/EC.txt >> tmp2.txt
cat tmp2.txt | sort | uniq -u | sort -R | head -n 20 >> to_keep.txt
# keeps 20 original images for TVPR dataset (exclude those that are converted to synthetic images)
cat dataset_list/EDE.txt | sort | uniq > tmp.txt
ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt
cat tmp.txt | sort | uniq -d > tmp2.txt
cat dataset_list/EDE.txt >> tmp2.txt
cat tmp2.txt | sort | uniq -u | sort -R | head -n 20 >> to_keep.txt
# keeps 20 original images for TVPR dataset (exclude those that are converted to synthetic images)
cat dataset_list/EDI.txt | sort | uniq > tmp.txt
ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt
cat tmp.txt | sort | uniq -d > tmp2.txt
cat dataset_list/EDI.txt >> tmp2.txt
cat tmp2.txt | sort | uniq -u | sort -R | head -n 20 >> to_keep.txt
# keeps 20 original images for TVPR dataset (exclude those that are converted to synthetic images)
cat dataset_list/EDOD.txt | sort | uniq > tmp.txt
ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt
cat tmp.txt | sort | uniq -d > tmp2.txt
cat dataset_list/EDOD.txt >> tmp2.txt
cat tmp2.txt | sort | uniq -u | sort -R | head -n 20 >> to_keep.txt
# keeps 20 original images for TVPR dataset (exclude those that are converted to synthetic images)
cat dataset_list/TVMPC.txt | sort | uniq > tmp.txt
ls synthetic_images/ | grep ".jpg" | sort | uniq >> tmp.txt
cat tmp.txt | sort | uniq -d > tmp2.txt
cat dataset_list/TVMPC.txt >> tmp2.txt
cat tmp2.txt | sort | uniq -u | sort -R | head -n 20 >> to_keep.txt

# converting xml annotations into one csv file -> output.csv
python xml_to_csv.py --include_aug 0
# if error occurs, quit program
if [ "$?" -ne 0 ]; then
		exit $?
fi
cat tmp.csv | tail -n +2 | sort -R > output.csv
rm tmp.csv
# if error occurs, quit program
if [ "$?" -ne 0 ]; then
		exit $?
fi

# preparing list of images
cat output.csv | cut -d "," -f1 | sort | uniq > tmp.txt
# images to exclude (EDOB and to_keep)
cat dataset_list/EDOB.txt | sort | uniq >> tmp.txt
cat to_keep.txt >> tmp.txt
# NOTE: if there are other images that you dont want to include in to_augment.txt, write the following code:
# cat xxx.txt | sort | uniq >> tmp.txt 		# change 'xxx' to the name of the textfile / list of images to exclude
cat tmp.txt | sort | uniq -u > to_augment.txt

# remove temperory files
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
