"""
===========
Ginput Demo
===========

This provides examples of uses of interactive functions, such as ginput,

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import glob, os, re
import pickle as pkl

png_list = sorted(glob.glob('mngs/*.chop.png'))
csv_list = sorted(glob.glob('mngs/*.chop.csv'))
file_set = set(png.rpartition('.png')[0] for png in png_list) - set(csv.rpartition('.csv')[0] for csv in csv_list)
file_list = sorted([file + '.png' for file in file_set])

for file in file_list:
	im = plt.imread(file)
	plt.imshow(im)
	x = plt.ginput(-1, timeout=300)
	pd.DataFrame(x).to_csv(file.rpartition('.')[0] + '.csv')

plt.close()

