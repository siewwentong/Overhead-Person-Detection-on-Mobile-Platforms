import pandas as pd

import sys

if (len(sys.argv) != 3):
	print("USAGE: python add_headers <csv_without_headers.csv> <output.csv>")
	exit()
in_file = sys.argv[1]
out_file = sys.argv[2]
df = pd.read_csv(in_file, header=None)
df.rename(columns={0: 'filename', 1: 'height', 2: 'width', 3: 'xmin', 4: 'ymin', 5: 'xmax', 6: 'ymax', 7: 'class'}, inplace=True)
df.to_csv(out_file, index=False) # save to new csv file
