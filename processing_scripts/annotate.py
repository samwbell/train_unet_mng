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

def write_file(path, text):
	with open(path, 'w') as path_stream:
		path_stream.write(text)
		path_stream.close()


for true_loc_file in sorted(glob.glob('handlocs/*.txt')):

	idnum = true_loc_file.rpartition('/')[2].rpartition('.txt')[0]

	true_locs = pd.read_table(true_loc_file)

	true_locs = true_locs.assign(X=(true_locs['X (um)']/0.2508).astype(int))
	true_locs = true_locs.assign(Y=(true_locs['Y (um)']/0.2508).astype(int))

	#fimage = cv2.imread('images/30159.png')

	file_list = sorted(glob.glob('cropped/' + idnum + '*.png'))

	false_positive_list = []
	false_negative_list = []
	match_list = []
	fulloutfile = ''
	for file in file_list:

		base = file.rpartition('/')[2].rpartition('.png')[0]

		cropped = cv2.imread(file)
		params = pkl.load(open('cropped/' + base + '.params.pkl', 'rb'))

		h,w = cropped.shape[:2]

		testis_locs = true_locs[(true_locs['X'] > params['x0']) & (true_locs['X'] < params['x1'])]
		testis_locs = testis_locs[(testis_locs['Y'] > params['y0']) & (testis_locs['Y'] < params['y1'])]

		testis_locs = testis_locs.assign(x=(true_locs['X'].astype(int) - params['x0']))
		testis_locs = testis_locs.assign(y=(true_locs['Y'].astype(int) - params['y0']))

		for row in testis_locs.itertuples():
			cv2.circle(cropped, (row.x,row.y), 7, (0,255,255), -1)

		cv2.imwrite('handlabeled/' + base + '.hand.png', cropped)



