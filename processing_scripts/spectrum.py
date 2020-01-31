#import openslide
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
import sys

#if len(sys.argv) < 2:
#	raise Exception('You need to input the image filename as a command line parameter like: python spectrum.py germsample1.png')
#file = str(sys.argv[1])

base = 'samples/mngsample'

mngfiles = glob.glob(base + '?.png')

print(mngfiles)

red_hists = []
green_hists = []
blue_hists = []

for file in mngfiles:
	ifile_base = file.partition('.tiff')[0]

	im = cv2.imread(file)
	red = im[:,:,2]
	green = im[:,:,1]
	blue = im[:,:,0]

	print(np.min(cv2.divide(red,green, scale=70.0)))
	print(np.mean(cv2.divide(red,green, scale=70.0)))
	print(np.max(cv2.divide(red,green, scale=70.0)))

	if True:
		gray = cv2.imread(file, 0)
		cv2.imwrite(ifile_base + '.gray.jpg', gray)

	if True:
		cv2.imwrite(ifile_base + '.bgratio.jpg', cv2.divide(blue,green, scale=100.0))
		cv2.imwrite(ifile_base + '.rgratio.jpg', cv2.add(-50, cv2.divide(red,green, scale=100.0)))

	ihist_red,bins = np.histogram(red.ravel(),256,[0,256])
	ihist_green,bins = np.histogram(green.ravel(),256,[0,256])
	ihist_blue,bins = np.histogram(blue.ravel(),256,[0,256])

	red_hists.append(ihist_red)
	green_hists.append(ihist_green)
	blue_hists.append(ihist_blue)

red_hist = np.mean(red_hists, axis=0)
green_hist = np.mean(green_hists, axis=0)
blue_hist = np.mean(blue_hists, axis=0)

fig = plt.figure()
plt.plot(red_hist,'r')
plt.plot(green_hist,'g')
plt.plot(blue_hist,'b')
plt.xlabel('Color Value')
plt.ylabel('Pixel Count')
file_base = base + '.mean'
fig.savefig(file_base + '.hist.pdf', bbox_inches='tight')
plt.close(fig)

