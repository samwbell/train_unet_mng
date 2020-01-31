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

def write_file(path, text):
	with open(path, 'w') as path_stream:
		path_stream.write(text)
		path_stream.close()

file_list = sorted(glob.glob('images/2*.svs'))


for file in file_list:

	base = file.rpartition('/')[2].rpartition('.svs')[0]
	print(base)

	if True: #not os.path.isfile('images/' + base + '.png'):
		print('Reading...')
		t1 = time.time()
		tslide = openslide.OpenSlide(file)
		tdimensions = tuple(x*0.2508 for x in tslide.dimensions)
		tdimensions = tslide.dimensions
		timage = tslide.get_thumbnail(tdimensions)
		timage.save('images/' + base + '.png')
		write_file('images/' + base + '.txt', str(tslide.properties))
		t2 = time.time()
		print('Done in ' + str(round(t2-t1,2)) + ' s')


	print('Cropping')
	t1 = time.time()
	raw = cv2.imread('images/' + base + '.png')
	gray = cv2.imread('images/' + base + '.png', 0)
	threshold = 200
	bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

	height,width = bw.shape[:2]


	#cv2.imwrite('bw.png', bw)

	bw_copy = bw.copy()

	contours, im2 = cv2.findContours(bw_copy,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	i = 1
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if cv2.contourArea(cnt) > height*width/100 and h*w < height*width/1.1:
			cv2.imwrite('cropped/' + base + '.' + str(i) + '.png', raw[int(y-h/200):int(y+h*201/200),int(x-w/200):int(x+w*201/200)])
			params={'y0':int(y-h/200), 'y1':int(y+h*201/200), 'x0':int(x-w/200), 'x1':int(x+w*201/200)}
			pkl.dump(params, open('cropped/' + base + '.' + str(i) + '.params.pkl','wb'))
			i += 1

	t2 = time.time()
	print('Done in ' + str(round(t2-t1,2)) + ' s')



