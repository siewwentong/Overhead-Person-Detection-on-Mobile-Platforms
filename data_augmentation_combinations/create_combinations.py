'''
This script takes in all data augmentations from "basic_op_txt" and creates combinations that has a maximum of "no_of_op" data augmentations
Then, it saves all the combinations created in output_txt. Data augmentations that are of the same family are not grouped.
Written by: Tong Siew Wen (January 2021)
...
How to use the code:
- python3 create_combinations.py --no_of_op <maximum number of operation in each combination> (add other parameters if needed)
...
What is required:
- a list (textfile) that has the name of the data augmentation techniques
	- data augmentations named by family followed by its type (e.g. Rotation90, not 90Rotation; Saturation_in, not in_Saturation)
		- the reason is the first three letters are used to identify same family and prevent their members to be grouped together
		- this also means the first three letters of the family name should be unique (e.g. abcd and abce are considered to be the same family)
- input parameter for --no_of_op. Refers to the maximum number of operation in each combination ( >= 1)
	- e.g. --no_of_op 3 will give a list that consists of operations with 1 to 3 data augmentaions per combination (per line in textfile)
...
To take note:
- if the input for "no_of_op" is greater than the maximum number of operation that can be grouped in a combination (n),
  the script will only create combinations with up to n datat augmentations per combination
  	- e.g. I have 3 families of data augmentation, maximum of 3 data augmentation per combination can be formed.
	       If I inputed --no_of_op 100, the script will only create combinations with up to 3 data augmentations per combination, instead of 100.
- All combinations formed are unique (no repetition)
- If you want to edit the code, it is suggested to make a copy of the code before doing so.
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--no_of_op", help = "maximum number of operation in each combination ( >= 1)")
parser.add_argument("--output_txt", default = "data_augmentation_combination.txt")
parser.add_argument("--basic_op_txt", default = "basic_operations.txt")
args = parser.parse_args()

# n = maximum number of operation in each combination
n = int(args.no_of_op)
# n must be > 1 (at least one data augmentation per combination)
if (n < 1):
	raise ValueError("maximum number of operation in each combination must be at least 1")

with open(args.basic_op_txt, 'r') as txt:
	# getting all basic operations
	operations = txt.readlines()
	with open(args.output_txt, 'w') as out:
		single = []
		# creating single operation per combination
		for operation in operations:
			single.append(operation[:-1])
			out.write(operation[:-1] + "\n")

		if n > 1:
			# creating two operations per combination
			two = []
			for i, operation1 in enumerate(single):
				combinations = []
				for operation2 in single[i+1:]:
					# if the data augmentations are of the same family, create combination
					if operation1[:3] != operation2[:3]:
						combinations.append(operation1 + " " + operation2)
						out.write(operation1 + " " + operation2 + "\n")
				two.append(combinations)
			old_pair = two
			n -= 1


			while n > 1:
				# creating three or more operations per combination
				new_pair = []
				for i, operation1 in enumerate(single):
					combinations = []
					for operations in old_pair[i+1:]:
						for operation2 in operations:
							# if the data augmentations are of the same family, create combination
							if operation1[:3] != operation2[:3]:
								combinations.append(operation1 + " " + operation2)
								out.write(operation1 + " " + operation2 + "\n")
							else:
								break
					new_pair.append(combinations)
				old_pair = new_pair
				n -= 1
