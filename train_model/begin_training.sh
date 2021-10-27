#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "USAGE: bash begin_training.sh <NAME OF TRAINED MODEL>"
		exit 1
fi

# This script basically runs the training script of tensorflow object_detection for you
# you just dont have to deal with the filepaths everytime you run the training
dir=$PWD
config_file="a.config"
delim="/"
TROUT="training_output"
cd models/research
python setup.py install
python object_detection/model_main.py \
    --logtostderr=true \
    --model_dir=$dir$delim$TROUT$delim$1$delim \
    --pipeline_config_path=$dir$delim$config_file
