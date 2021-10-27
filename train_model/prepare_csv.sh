#!/bin/bash

# This script split training and validation data, add header to train.csv and val.csv,
# generate tfrecords for training and validation data, and edit the config file.
# Edited by: Tong Siew Wen (December 2020)
# ...
# What is required:
# - a csv file that contains all the annotations
# - the three python scripts should be in the same directory with this shell script
# 		- split_train_and_val.py, generate_tfrecords.py and pipiline_editor.py
# - others: see "What is required" in split_train_and_val.py
# ...
# To take note:
# - This script deletes final_train and final_val folder everytime it runs. If you need the previous one, please rename them.
# - This script overwrites train.record and val.record everytime it runs. If you need the previous one, please rename them.


# check whether there are two parameters inputed
if [ "$#" -ne 2 ]; then
    echo "USAGE: bash prepare_csv.sh <csv filename> <synthetic (True:1/False:0)>"
		exit 1
fi

# adding python paths
export PYTHONPATH="${PYTHONPATH}:${PWD}/models/"
export PYTHONPATH="${PYTHONPATH}:${PWD}/models/research/"
export PYTHONPATH="${PYTHONPATH}:${PWD}/models/research/slim/"
export PYTHONPATH="${PYTHONPATH}:${PWD}/models/research/object_detection/utils/"
export PYTHONPATH="${PYTHONPATH}:${PWD}/models/research/object_detection/"

rm -rf final_val/
# if return value is non zero (have error), end program
if [ "$?" -ne 0 ]; then
		exit $?
fi
rm -rf final_train/
# if return value is non zero (have error), end program
if [ "$?" -ne 0 ]; then
		exit $?
fi

# create list for validation set
cat dataset_list/EDOB.txt | sort -R > tmp.txt

# split training and validation data
python split_train_and_val.py --validation_file tmp.txt --csv_input $1 --synthetic $2
# if return value is non zero (have error), end program
if [ "$?" -ne 0 ]; then
		exit $?
fi

# adding headers for train.csv anf val.csv
python add_headers.py final_train/train.csv final_train/train_h.csv
if [ "$?" -ne 0 ]; then
		exit $?
fi
python add_headers.py final_val/val.csv final_val/val_h.csv
if [ "$?" -ne 0 ]; then
		exit $?
fi
rm final_train/train.csv
if [ "$?" -ne 0 ]; then
		exit $?
fi
rm final_val/val.csv
if [ "$?" -ne 0 ]; then
		exit $?
fi
rm tmp.txt
if [ "$?" -ne 0 ]; then
		exit $?
fi
mv final_val/val_h.csv final_val/val.csv
if [ "$?" -ne 0 ]; then
		exit $?
fi
mv final_train/train_h.csv final_train/train.csv
if [ "$?" -ne 0 ]; then
		exit $?
fi

# generating tfrecords for training and validation
python generate_tfrecord.py --csv_input final_train/train.csv --output_path records/train.record --image_dir final_train/images/
if [ "$?" -ne 0 ]; then
		exit $?
fi
python generate_tfrecord.py --csv_input final_val/val.csv --output_path records/val.record --image_dir final_val/images/
if [ "$?" -ne 0 ]; then
		exit $?
fi

# edit a.config
python pipeline_editor.py
