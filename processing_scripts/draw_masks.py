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

file_list = sorted(glob.glob('mngs/2*.chop.csv'), key=naturalSort)

if True:
	for file in file_list:

		#print(file.rpartition('.chop.csv')[0])

		points = pd.read_csv(file)

		im = cv2.imread(file.rpartition('.csv')[0] + '.png')

		cnt = np.array(points[['0','1']].values.tolist()).reshape((-1,1,2)).astype(np.int32)

		cv2.drawContours(im,[cnt],0,(255,255,255),1)

		cv2.imwrite(file.rpartition('.chop.csv')[0] + '.mask.png',im)

loc_file_list = sorted(glob.glob('testislocs/2*.locs.pkl'), key=naturalSort)

for loc_file in loc_file_list:
	print(loc_file)
	locs = pkl.load(open(loc_file, 'rb'))
	if locs.shape[0] != 0:
		base = loc_file.rpartition('.locs.pkl')[0].rpartition('/')[2]
		print(base)
		im = cv2.imread('cropped/' + base + '.png')
		h,w = im.shape[:2]
		blank = np.zeros((h,w),np.uint8)
		i = 0
		for row in locs.itertuples():
			file_list = sorted(glob.glob('mngs/' + base + '.' + str(i) + '.chop.csv'), key=naturalSort)
			i += 1
			for file in file_list:
				points = pd.read_csv(file)
				cnt = np.array(points[['0','1']].values.tolist()).reshape((-1,1,2)).astype(np.int32)
				cnt = np.array([[row.x - 100,row.y - 100]]) + cnt
				cv2.drawContours(im,[cnt],0,(0,255,0),2)
				cv2.drawContours(blank,[cnt],0,255,-1)
				cv2.drawContours(blank,[cnt],0,0,2)
	else:
		base = loc_file.rpartition('.locs.pkl')[0].rpartition('/')[2]
		print(base)
		im = cv2.imread('cropped/' + base + '.png')
		h,w = im.shape[:2]
		blank = np.zeros((h,w),np.uint8)
	cv2.imwrite('masks/' + base + '.masked.png',im)
	cv2.imwrite('masks/' + base + '.mask.png',blank)

