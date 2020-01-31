#import openslide
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

true_loc_file_list = sorted(glob.glob('handlocs/30159*.txt'), key=naturalSort)

master_false_positive_list = []
master_false_negative_list = []
master_match_list = []

for true_loc_file in true_loc_file_list:

	image_num = true_loc_file.rpartition('/')[2].rpartition('.txt')[0]

	true_locs = pd.read_table(true_loc_file)

	true_locs = true_locs.assign(X=(true_locs['X (um)']).astype(int))
	true_locs = true_locs.assign(Y=(true_locs['Y (um)']).astype(int))

	#fimage = cv2.imread('images/30159.png')

	file_list = sorted(glob.glob('replication/' + image_num + '.?.5.tif'))

	print(file_list)

	false_positive_list = []
	false_negative_list = []
	match_list = []
	fulloutfile = ''
	for file in file_list:

		base = file.rpartition('/')[2].rpartition('.tif')[0].rpartition('.')[0]

		print(base)

		filtered = cv2.imread(file)
		filtered_bw = cv2.imread(file, 0)
		params = pkl.load(open('replication/' + base + '.params.pkl', 'rb'))

		h,w = filtered.shape[:2]

		cv2.rectangle(filtered,(0,0),(w,h), (255,255,255),-1)

		testis_locs = true_locs[(true_locs['X'] > params['x0']) & (true_locs['X'] < params['x1'])]
		testis_locs = testis_locs[(testis_locs['Y'] > params['y0']) & (testis_locs['Y'] < params['y1'])]

		testis_locs = testis_locs.assign(x=(true_locs['X'].astype(int) - params['x0']))
		testis_locs = testis_locs.assign(y=(true_locs['Y'].astype(int) - params['y0']))

		def matchQ(cx,cy):
			for row in testis_locs.itertuples():
				dist = math.sqrt((row.x - cx)**2 + (row.y - cy)**2)
				if dist <= 15:
					return True
			return False

		false_positives = 0
		cxs = []
		cys = []
		contours, im2  = cv2.findContours(filtered_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area < 0.9*h*w or area < 50:
				M = cv2.moments(cnt)
				#cv2.drawContours(filtered,[cnt],0,(0,0,0),2)
				if M['m00'] != 0 and area > 10:
					cx = int(M['m10']/M['m00'])
					cy = int(M['m01']/M['m00'])
					cxs.append(cx)
					cys.append(cy)	
					if matchQ(cx,cy):
						cv2.drawContours(filtered,[cnt],0,(0,0,0),-1)
					else:
						cv2.drawContours(filtered,[cnt],0,(255,0,0),-1)
						#print(cx,cy)
						false_positives += 1
						#print(false_positives)
						#cv2.circle(filtered, (cx,cy), 10, (0,255,255), -1)

					#cv2.circle(filtered, (cx,cy), 5, (0,255,255), -1)

		def loc_matchQ(x,y):
			for i in list(range(len(cxs))):
				dist = math.sqrt((x - cxs[i])**2 + (y - cys[i])**2)
				if dist <= 15:
					return True
			return False

		#masked = cv2.imread('cropped/' + base + '.masked.jpg')
		#smalldot = masked.copy()

		matches = 0
		false_negatives = 0	
		for row in testis_locs.itertuples():

			#cv2.circle(masked, (row.x,row.y), 15, (0,255,0), -1)

			#print(row.x,row.y)
			if loc_matchQ(row.x,row.y):
				cv2.circle(filtered, (row.x,row.y), 15, (0,255,0), 2)
				#cv2.circle(smalldot, (row.x,row.y), 7, (0,255,0), -1)
				matches += 1
			else:
				cv2.circle(filtered, (row.x,row.y), 15, (0,0,255), 2)
				#cv2.circle(smalldot, (row.x,row.y), 9, (0,255,255), -1)
				#cv2.circle(smalldot, (row.x,row.y), 7, (0,0,255), -1)
				false_negatives += 1

			#cv2.circle(fimage, (x,y), 50, (0,255,255), 5)

		print(matches,false_positives,false_negatives)
		if matches == 0:
			precision = 0
			recall = 0
			f1 = 0
		else:
			precision = float(matches)/float(matches + false_positives)
			recall = float(matches)/float(matches + false_negatives)
			f1 = 2*precision*recall/float(precision + recall)

		match_list.append(matches)
		false_negative_list.append(false_negatives)
		false_positive_list.append(false_positives)

		outfile = 'Image ' + base + ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
				'\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'
		fulloutfile += outfile
		write_file('replication/' + base + '.results.txt', outfile)
		#print(outfile)

		#cv2.imwrite('30159.labeled.png', fimage)
		cv2.imwrite('replication/' + base + '.labeled.png', filtered)

		#cv2.imwrite('replication/' + base + '.masked.labeled.png', masked)

		#cv2.imwrite('replication/' + base + '.masked.smalldot.png', smalldot)

	matches = sum(match_list)
	false_negatives = sum(false_negative_list)
	false_positives = sum(false_positive_list)

	master_match_list.append(matches)
	master_false_negative_list.append(false_negatives)
	master_false_positive_list.append(false_positives)

	if matches == 0:
			precision = 0
			recall = 0
			f1 = 0
	else:
		precision = float(matches)/float(matches + false_positives)
		recall = float(matches)/float(matches + false_negatives)
		f1 = 2*precision*recall/float(precision + recall)

	fulloutfile += 'Total for ' + image_num + ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
				'\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3))
	write_file('replication/' + image_num + '.results.txt', fulloutfile)
	print(fulloutfile)


matches = sum(master_match_list)
false_negatives = sum(master_false_negative_list)
false_positives = sum(master_false_positive_list)

if matches == 0:
		precision = 0
		recall = 0
		f1 = 0
else:
	precision = float(matches)/float(matches + false_positives)
	recall = float(matches)/float(matches + false_negatives)
	f1 = 2*precision*recall/float(precision + recall)

finaloutfile = 'Final results:\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
				'\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3))

write_file('replication/final.results.txt', finaloutfile)
print(finaloutfile)





