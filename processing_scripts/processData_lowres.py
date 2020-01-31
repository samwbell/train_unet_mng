import openslide
import glob, os, re
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import time
from multiprocessing import Pool
import shutil
import pickle as pkl
import math

def circularity(cnt):
	return 4 * math.pi * cv2.contourArea(cnt) / cv2.arcLength(cnt, True)**2



def write_binary(file, downsize=1.0):
	base = file.rpartition('/')[2].rpartition('.png')[0]

	raw = cv2.imread(file)

	height,width = raw.shape[:2]

	color = cv2.resize(raw, (0,0), fx=1/float(downsize), fy=1/float(downsize)) 

	if downsize != 1.0:
		cv2.imwrite(file.partition('.png')[0] + '.' + str(int(downsize)) + 'x.jpg', color)

	cf = 167.4220504227964/color.mean()
	rescaled = (color.astype(np.float)*cf).astype(np.uint8)

	cv2.imwrite('cropped/' + base + '.rescaled.jpg', rescaled)


	lower = np.array([140,80,90])
	upper = np.array([190,150,150])

	bw = cv2.inRange(rescaled,lower,upper)

	dlower = np.array([50,25,25])
	dupper = np.array([125,100,100])

	blower = np.array([205,205,205])
	bupper = np.array([255,255,255])

	dbw = cv2.inRange(rescaled,dlower,dupper)
	contours, im2  = cv2.findContours(dbw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		area = float(cv2.contourArea(cnt))
		if area > 25:
			cv2.drawContours(dbw,[cnt],0,255,35)

	cv2.imwrite('binary/' + base + '.dark_cells.jpg',cv2.bitwise_not(dbw))

	bbw = cv2.inRange(rescaled,blower,bupper)
	contours, im2  = cv2.findContours(bbw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		area = float(cv2.contourArea(cnt))
		if area > 0.05*width*height:
			cv2.drawContours(dbw,[cnt],0,255,300)
		elif area > 25:
			cv2.drawContours(dbw,[cnt],0,255,-1)
			#cv2.drawContours(dbw,[cnt],0,255,10)

	cv2.imwrite('binary/' + base + '.dark_and_light_cells.jpg',cv2.bitwise_not(dbw))

	contours, im2  = cv2.findContours(dbw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		area = float(cv2.contourArea(cnt))
		if area < 5000:
			cv2.drawContours(dbw,[cnt],0,0,-1)
		else:
			cv2.drawContours(dbw,[cnt],0,255,10)

	cv2.imwrite('binary/' + base + '.dark_filled.jpg',cv2.bitwise_not(dbw))

	contours, im2  = cv2.findContours(cv2.bitwise_not(dbw),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		area = float(cv2.contourArea(cnt))
		if area < 15000:
			cv2.drawContours(dbw,[cnt],0,255,-1)
		else:
			cv2.drawContours(dbw,[cnt],0,0,5)

	cv2.imwrite('binary/' + base + '.dark_filled_expanded.jpg',cv2.bitwise_not(dbw))

	if not os.path.isdir('binary'):
		os.mkdir('binary')

	cv2.imwrite('binary/' + base + '.png',cv2.bitwise_not(cv2.bitwise_and(bw, cv2.bitwise_not(dbw))))
	cv2.imwrite('cropped/' + base + '.masked.jpg',cv2.bitwise_and(raw, raw, mask=cv2.bitwise_not(dbw)))
	cv2.imwrite('binary/' + base + '.dark.jpg',cv2.bitwise_not(dbw))

def fill_holes(file):
	base = file.rpartition('/')[2].rpartition('.png')[0]

	#print(base)

	bw = cv2.bitwise_not(cv2.imread(file, 0))

	#bw = cv2.medianBlur(bw,3)
	#for i in range(5):
		#blurred = cv2.medianBlur(blurred,3)

	#cv2.imwrite(file.partition('.png')[0] + '.blurred.jpg', cv2.bitwise_not(blurred))

	height,width = bw.shape[:2]
	fbw = bw.copy()
	cv2.rectangle(fbw,(0,0),(width,height), 0,-1)
	#cv2.imwrite('test.png',cv2.bitwise_not(fbw))
	#exit()

	contours, im2  = cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		area = float(cv2.contourArea(cnt))
		if area > 25 and area/h/w > 0.1:
				cv2.drawContours(fbw,[cnt],0,255,2)

	contours, im2  = cv2.findContours(fbw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

	cv2.imwrite(file.partition('.png')[0] + '.rawblurred.jpg', cv2.bitwise_not(fbw))

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		area = float(cv2.contourArea(cnt))
		if (h * w) < 700 or area < 750:
			cv2.drawContours(fbw,[cnt],0,0,-1)	


	cv2.imwrite(file.partition('.png')[0] + '.blurred.jpg', cv2.bitwise_not(fbw))

	contours, im2  = cv2.findContours(cv2.bitwise_not(fbw),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	large_cnts = []
	for cnt in contours:
		if cv2.contourArea(cnt) < 300 or (cv2.contourArea(cnt) < 1500 and circularity(cnt) < 0.5):
			cv2.drawContours(fbw,[cnt],0,255,-1)
		elif cv2.contourArea(cnt) > 0.1*height*width:
			large_cnts.append(cnt)
	for large_cnt in large_cnts:
		cv2.drawContours(fbw,[large_cnt],0,0,5)

	cv2.imwrite(file.partition('.png')[0] + '.filled.jpg', cv2.bitwise_not(fbw))	

	#cv2.rectangle(fbw,(0,0),(width,height), 255,3)
	contours, im2 = cv2.findContours(fbw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if False: #cv2.contourArea(cnt) > 0.1*height*width:
			hull = cv2.convexHull(cnt, returnPoints = False)
			#epsilon = 0.001*cv2.arcLength(cnt,True)
			#approx = cv2.approxPolyDP(cnt,epsilon,True)
			defects = cv2.convexityDefects(cnt,hull)
			fars = []
			for defect in defects:
				s,e,f,d = defect[0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				fars.append(far)
				cv2.circle(fbw,far,5,0,-1)
			#for far in fars:
				#x,y = far
				#nearest = sorted([[math.sqrt((fari[0] - x)**2 + (fari[1] - y)**2),fari] for fari in fars])[1]
				#if nearest[0] < 15:
					#print(nearest)
					#cv2.circle(fbw,far,5,[0,0,255],-1)
			#cv2.drawContours(fbw,[cnt],0,0,7)
	#cv2.rectangle(fbw,(0,0),(width,height), 0,3)

	contours, im2 = cv2.findContours(fbw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		cv2.drawContours(fbw,[cnt],0,0,7)

	if not os.path.isdir('filled'):
		os.mkdir('filled')

	cv2.imwrite('filled/' + base + '.png', cv2.bitwise_not(fbw))

def despeckle(file):
	base = file.rpartition('/')[2].rpartition('.png')[0]

	despeckled = cv2.medianBlur(cv2.imread(file, 0),1)

	if not os.path.isdir('despeckled'):
		os.mkdir('despeckled')

	cv2.imwrite('despeckled/' + base + '.png', despeckled)

def filter(file, size_cutoff, circularity_cutoff):
	base = file.rpartition('/')[2].rpartition('.png')[0]

	bw = cv2.bitwise_not(cv2.imread(file, 0))
	height,width = bw.shape[:2]
	filtered = np.zeros((height,width), np.uint8)

	contours, im2 = cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

	for cnt in contours:
		if cv2.contourArea(cnt) >= size_cutoff and circularity(cnt) >= circularity_cutoff:
			cv2.drawContours(filtered,[cnt],0,255,-1)

	if not os.path.isdir('filtered'):
		os.mkdir('filtered')

	cv2.imwrite('filtered/' + base + '.png', cv2.bitwise_not(filtered))

downsize=1

if True:
	print('Converting to binary...')
	t1 = time.time()
	file_list = sorted(glob.glob('cropped/*.png'))
	for file in file_list:
		write_binary(file, downsize=downsize)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

if True:
	print('Filling holes...')
	t1 = time.time()
	file_list = sorted(glob.glob('binary/*.png'))
	for file in file_list:
		fill_holes(file)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

if True:
	print('Despeckling (deprecated step)...')
	t1 = time.time()
	file_list = sorted(glob.glob('filled/*.png'))
	for file in file_list:
		despeckle(file)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')

if True:
	print('Filtering...')
	t1 = time.time()
	file_list = sorted(glob.glob('despeckled/*.png'))
	for file in file_list:
		filter(file, 2623.2/1.5/downsize**2, 0.25)
	t2 = time.time()
	print('Done in: ' + str(round(t2-t1, 3)) + ' s')






