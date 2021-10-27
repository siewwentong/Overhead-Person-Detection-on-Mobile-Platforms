import cv2
import glob
import random
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
args = parser.parse_args()

images = glob.glob(args.input_dir + "*.jpg")
random.shuffle(images)
print("total images: ", len(images))

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for idx, image in enumerate(images):
	im = cv2.imread(image)
	im = cv2.resize(im, (256,256))
	try:
		cv2.imwrite(args.output_dir + str(idx) + ".jpg", im)
	except:
		print(image)
