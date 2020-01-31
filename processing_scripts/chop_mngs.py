import openslide
import glob, os, re
import cv2
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import time
from multiprocessing import Pool
import shutil
import pickle as pkl
import math

"""
Sorts based on natural ordering of numbers, ie. "12" > "2" 
"""
def naturalSort(String_): 
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

"""
Writes a string to a file
"""
def write_file(path, text):
	with open(path, 'w') as path_stream:
		path_stream.write(text)
		path_stream.close()

true_loc_file_list = sorted(glob.glob('handlocs/27*.txt'), key=naturalSort)

for true_loc_file in true_loc_file_list:

	image_num = true_loc_file.rpartition('/')[2].rpartition('.txt')[0]

	true_locs = pd.read_csv(true_loc_file, sep='\t')

	true_locs = true_locs.assign(X=(true_locs['X (um)']/0.2508).astype(int))
	true_locs = true_locs.assign(Y=(true_locs['Y (um)']/0.2508).astype(int))

	#fimage = cv2.imread('images/30159.png')

	file_list = sorted(glob.glob('cropped/' + image_num + '.?.png'))

	for file in file_list:

		base = file.rpartition('/')[2].rpartition('.png')[0]

		print(base)

		cropped = cv2.imread(file)

		params = pkl.load(open('cropped/' + base + '.params.pkl', 'rb'))

		testis_locs = true_locs[(true_locs['X'] > params['x0']) & (true_locs['X'] < params['x1']) &\
			(true_locs['Y'] > params['y0']) & (true_locs['Y'] < params['y1'])]

		testis_locs = testis_locs.assign(x=(testis_locs['X'].astype(int) - params['x0']))
		testis_locs = testis_locs.assign(y=(testis_locs['Y'].astype(int) - params['y0']))

		testis_locs.to_pickle('testislocs/' + base + '.locs.pkl')
		testis_locs.to_csv('testislocs/' + base + '.locs.csv')


		if False:
			count = 0
			if testis_locs.shape[0] != 0:
				for row in testis_locs.itertuples():
					chop_im = cropped[row.y - 100:row.y + 100,row.x - 100:row.x + 100]
					cv2.imwrite('mngs/' + base + '.' + str(count) + '.chop.png', chop_im)
					count += 1




