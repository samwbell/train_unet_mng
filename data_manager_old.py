import os, glob, shutil
import numpy as np
import pandas as pd
import cv2
import pickle as pkl
import random
from random import randint
from random import shuffle
from itertools import combinations
from collections import OrderedDict
from operator import itemgetter
import time

random.seed(1)
np.random.seed(1)
tl_files = sorted(glob.glob('testislocs/*.locs.csv'))
panels = set(fname.rpartition('/')[2].partition('.')[0] for fname in tl_files)

# Old fold partitioning scheme
if False:

	n_mngs_dict = {panel:sum([pd.read_csv(loc_file).shape[0] for loc_file in \
		glob.glob('testislocs/' + panel + '*.locs.csv')]) for panel in panels}

	folds = []
	folded = set()

	sorted_panels = sorted(list(panels), key=lambda x:n_mngs_dict[x])
	folds.append(set(sorted_panels[:4] + [sorted_panels[-1]]))
	folded.update(set(sorted_panels[:4] + [sorted_panels[-1]]))

	p_mean = sum([n_mngs_dict[panel] for panel in panels-folded])/4


	t1 = time.time()
	for i in range(4):
		if i < 2:
			n = 6
		else:
			n=5
		citer = combinations(panels-folded,n)
		diff = 99999
		best_combination = []
		for combination in citer:
			c_mean = sum([n_mngs_dict[panel] for panel in combination])
			if abs(c_mean - p_mean) < diff:
				best_combination = combination
				diff = abs(c_mean - p_mean)
		folds.append(set(best_combination))
		folded.update(set(best_combination))

	t2 = time.time()
	print(t2-t1)

	print(folds)
	print([[n_mngs_dict[panel] for panel in fold] for fold in folds])
	print([sum([n_mngs_dict[panel] for panel in fold]) for fold in folds])

folds = [{'30156', '30190', '27670', '27918', '27908'}, \
	{'30188', '30226', '30157', '27914', '30187', '30159'}, \
	{'27681', '27675', '30158', '30224', '27906', '30192'}, \
	{'30227', '27677', '27912', '30223', '27915'}, \
	{'27676', '27679', '27669', '30193', '30167'}]

test_set = folds[1]
holdout_set = folds[2]
train_set = folds[0]|folds[3]|folds[4]

def rotate(img,angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (h, w))
    return rotated

def augment(img):
    out_imgs = []
    for angle in [0,90,180,270]:
        rotated = rotate(img, angle)
        out_imgs.append(rotated)
        if angle < 100:
            out_imgs.append(cv2.flip(rotated, 0))
            out_imgs.append(cv2.flip(rotated, 1))
    return out_imgs

def specific_augment(img,i):
	aug_n = i%8
	angles = [0,90,180,270]
	if aug_n < 4:
		return rotate(img, angles[aug_n])
	elif aug_n == 4:
		rotated = rotate(img, 0)
		return cv2.flip(rotated, 0)
	elif aug_n == 5:
		rotated = rotate(img, 90)
		return cv2.flip(rotated, 0)
	elif aug_n == 6:
		rotated = rotate(img, 0)
		return cv2.flip(rotated, 1)
	elif aug_n == 7:
		rotated = rotate(img, 90)
		return cv2.flip(rotated, 1)

def get_training_data(n_mng = 20, n_random = 0):
	random.seed(1)

	if os.path.exists('data/membrane/raw_train'):
	    shutil.rmtree('data/membrane/raw_train')
	os.mkdir('data/membrane/raw_train')
	for panel in train_set:
	    loc_file_list = sorted(glob.glob('testislocs/' + panel + '*.locs.csv'))

	    for loc_file in loc_file_list:
	        #print(loc_file)
	        locs = pd.read_csv(loc_file)
	        base = loc_file.rpartition('.locs.csv')[0].rpartition('/')[2]
	        #print(base)
	        imx = cv2.imread('cropped/' + base + '.png',0)
	        imy = cv2.imread('masks/' + base + '.mask.png',0)
	        h,w = imx.shape[:2]
	        if locs.shape[0] != 0:
	            i = 0
	            for row in locs.itertuples():
	                for count in list(range(n_mng)):
	                    x_shift = randint(-255,255)
	                    y_shift = randint(-255,255)
	                    left = row.x - 256 + x_shift
	                    right = row.x + 256 + x_shift
	                    top = row.y - 256 + y_shift
	                    bottom = row.y + 256 + y_shift
	                    if (left > 0) & (top > 0) & (right < w) & (bottom < h):
		                    data_img = specific_augment(imx[top:bottom,left:right], i)
		                    label_img = specific_augment(imy[top:bottom,left:right], i)
		                    cv2.imwrite('data/membrane/raw_train/' + base + '.' + str(i) + '.' + str(count) + '.data.png',data_img)
		                    cv2.imwrite('data/membrane/raw_train/' + base + '.' + str(i) + '.' + str(count) + '.label.png',label_img)
	                i += 1
	        for i in list(range(n_random)):
	            x = randint(256,w-257)
	            y = randint(256,h-257)
	            left = x - 256
	            right = x + 256
	            top = y - 256
	            bottom = y + 256
	            cv2.imwrite('data/membrane/raw_train/' + base + '.random.' + str(i) + '.data.png',imx[top:bottom,left:right])
	            cv2.imwrite('data/membrane/raw_train/' + base + '.random.' + str(i) + '.label.png',imy[top:bottom,left:right])

	x_train_filenames = sorted(glob.glob('data/membrane/raw_train/*.data.png'))
	y_train_filenames = sorted(glob.glob('data/membrane/raw_train/*.label.png'))

	indices = list(range(len(x_train_filenames)))
	shuffle(indices)
	if os.path.exists('data/membrane/train'):
	    shutil.rmtree('data/membrane/train')
	os.mkdir('data/membrane/train')
	os.mkdir('data/membrane/train/image')
	os.mkdir('data/membrane/train/label')
	count = 0
	for i in indices:
	    cv2.imwrite('data/membrane/train/image/' + str(count) + '.png', cv2.imread(x_train_filenames[i],0))
	    cv2.imwrite('data/membrane/train/label/' + str(count) + '.png', cv2.imread(y_train_filenames[i],0))
	    count += 1
	return count 


def get_training_data_full_rotation(n_mng = 20, n_random = 0):
	random.seed(1)

	if os.path.exists('data/membrane/raw_train'):
	    shutil.rmtree('data/membrane/raw_train')
	os.mkdir('data/membrane/raw_train')
	for panel in train_set:
	    loc_file_list = sorted(glob.glob('testislocs/' + panel + '*.locs.csv'))

	    for loc_file in loc_file_list:
	        #print(loc_file)
	        locs = pd.read_csv(loc_file)
	        base = loc_file.rpartition('.locs.csv')[0].rpartition('/')[2]
	        #print(base)
	        imx = cv2.imread('cropped/' + base + '.png',0)
	        imy = cv2.imread('masks/' + base + '.mask.png',0)
	        h,w = imx.shape[:2]
	        if locs.shape[0] != 0:
	            i = 0
	            for row in locs.itertuples():
	                for count in list(range(n_mng)):
	                    x_shift = randint(-255,255)
	                    y_shift = randint(-255,255)
	                    left = row.x - 256 + x_shift
	                    right = row.x + 256 + x_shift
	                    top = row.y - 256 + y_shift
	                    bottom = row.y + 256 + y_shift
	                    if (left > 0) & (top > 0) & (right < w) & (bottom < h):
	                        data_imgs = augment(imx[top:bottom,left:right])
	                        label_imgs = augment(imy[top:bottom,left:right])
	                        for ai in list(range(len(data_imgs))):
	                            cv2.imwrite('data/membrane/raw_train/' + base + '.' + str(i) + '.' + str(count) + '.' + str(ai)  + '.data.png',data_imgs[ai])
	                            cv2.imwrite('data/membrane/raw_train/' + base + '.' + str(i) + '.' + str(count) + '.' + str(ai)  + '.label.png',label_imgs[ai])
	                i += 1
	        for i in list(range(n_random)):
	            x = randint(256,w-257)
	            y = randint(256,h-257)
	            left = x - 256
	            right = x + 256
	            top = y - 256
	            bottom = y + 256
	            cv2.imwrite('data/membrane/raw_train/' + base + '.random.' + str(i) + '.data.png',imx[top:bottom,left:right])
	            cv2.imwrite('data/membrane/raw_train/' + base + '.random.' + str(i) + '.label.png',imy[top:bottom,left:right])

	x_train_filenames = sorted(glob.glob('data/membrane/raw_train/*.data.png'))
	y_train_filenames = sorted(glob.glob('data/membrane/raw_train/*.label.png'))

	indices = list(range(len(x_train_filenames)))
	shuffle(indices)
	if os.path.exists('data/membrane/train'):
	    shutil.rmtree('data/membrane/train')
	os.mkdir('data/membrane/train')
	os.mkdir('data/membrane/train/image')
	os.mkdir('data/membrane/train/label')
	count = 0
	for i in indices:
	    cv2.imwrite('data/membrane/train/image/' + str(count) + '.png', cv2.imread(x_train_filenames[i],0))
	    cv2.imwrite('data/membrane/train/label/' + str(count) + '.png', cv2.imread(y_train_filenames[i],0))
	    count += 1
	return count 

def make_test_panels():
	random.seed(1)
	if os.path.exists('data/membrane/raw-test'):
	    shutil.rmtree('data/membrane/raw-test')
	os.mkdir('data/membrane/raw-test')
	for panel in test_set:
	    loc_file_list = sorted(glob.glob('testislocs/' + panel + '*.locs.csv'))

	    for loc_file in loc_file_list:
	        #print(loc_file)
	        locs = pd.read_csv(loc_file)
	        base = loc_file.rpartition('.locs.csv')[0].rpartition('/')[2]
	        #print(base)
	        imx = cv2.imread('cropped/' + base + '.png',0)
	        imy = cv2.imread('masks/' + base + '.mask.png',0)
	        h,w = imx.shape[:2]
	        step = 310
	        nh = (h-512)//step + 1
	        nw = (w-512)//step + 1
	        yedges = list(range(0,h-512,(h-512)//nh))
	        xedges = list(range(0,w-512,(w-512)//nw))
	        i = 0
	        for x in xedges:
	            for y in yedges:
	                left = x
	                right = x + 512
	                top = y
	                bottom = y + 512
	                cv2.imwrite('data/membrane/raw-test/' + base + '.random.' + str(i) + '.data.png',imx[top:bottom,left:right])
	                cv2.imwrite('data/membrane/raw-test/' + base + '.random.' + str(i) + '.label.png',imy[top:bottom,left:right])
	                params = {'x':x,'y':y,'i':i,'base':base}
	                pkl.dump(params,open('data/membrane/raw-test/' + base + '.random.' + str(i) + '.params.pkl', 'wb'))
	                i += 1
	x_test_filenames = sorted(glob.glob('data/membrane/raw-test/*.data.png'))
	y_test_filenames = sorted(glob.glob('data/membrane/raw-test/*.label.png'))
	params_test_filenames = sorted(glob.glob('data/membrane/raw-test/*.params.pkl'))
	indices = list(range(len(x_test_filenames)))
	shuffle(indices)
	if os.path.exists('data/membrane/test'):
	    shutil.rmtree('data/membrane/test')
	os.mkdir('data/membrane/test')
	os.mkdir('data/membrane/test/test')
	os.mkdir('data/membrane/test/test_label')
	count = 0
	for i in indices:
	    cv2.imwrite('data/membrane/test/test/' + str(count) + '.png', cv2.imread(x_test_filenames[i],0))
	    shutil.copyfile(params_test_filenames[i],'data/membrane/test/test/' + str(count) + '.params.pkl')
	    cv2.imwrite('data/membrane/test/test_label/' + str(count) + '.png', cv2.imread(y_test_filenames[i],0))
	    count += 1


def make_holdout_panels():
	random.seed(1)
	if os.path.exists('data/membrane/raw-holdout'):
	    shutil.rmtree('data/membrane/raw-holdout')
	os.mkdir('data/membrane/raw-holdout')
	for panel in holdout_set:
	    loc_file_list = sorted(glob.glob('testislocs/' + panel + '*.locs.csv'))

	    for loc_file in loc_file_list:
	        #print(loc_file)
	        locs = pd.read_csv(loc_file)
	        base = loc_file.rpartition('.locs.csv')[0].rpartition('/')[2]
	        #print(base)
	        imx = cv2.imread('cropped/' + base + '.png',0)
	        imy = cv2.imread('masks/' + base + '.mask.png',0)
	        h,w = imx.shape[:2]
	        step = 310
	        nh = (h-512)//step + 1
	        nw = (w-512)//step + 1
	        yedges = list(range(0,h-512,(h-512)//nh))
	        xedges = list(range(0,w-512,(w-512)//nw))
	        i = 0
	        for x in xedges:
	            for y in yedges:
	                left = x
	                right = x + 512
	                top = y
	                bottom = y + 512
	                cv2.imwrite('data/membrane/raw-holdout/' + base + '.random.' + str(i) + '.data.png',imx[top:bottom,left:right])
	                cv2.imwrite('data/membrane/raw-holdout/' + base + '.random.' + str(i) + '.label.png',imy[top:bottom,left:right])
	                params = {'x':x,'y':y,'i':i,'base':base}
	                pkl.dump(params,open('data/membrane/raw-holdout/' + base + '.random.' + str(i) + '.params.pkl', 'wb'))
	                i += 1
	x_holdout_filenames = sorted(glob.glob('data/membrane/raw-holdout/*.data.png'))
	y_holdout_filenames = sorted(glob.glob('data/membrane/raw-holdout/*.label.png'))
	params_holdout_filenames = sorted(glob.glob('data/membrane/raw-holdout/*.params.pkl'))
	indices = list(range(len(x_holdout_filenames)))
	shuffle(indices)
	if os.path.exists('data/membrane/holdout'):
	    shutil.rmtree('data/membrane/holdout')
	os.mkdir('data/membrane/holdout')
	os.mkdir('data/membrane/holdout/holdout')
	os.mkdir('data/membrane/holdout/holdout_label')
	count = 0
	for i in indices:
	    cv2.imwrite('data/membrane/holdout/holdout/' + str(count) + '.png', cv2.imread(x_holdout_filenames[i],0))
	    shutil.copyfile(params_holdout_filenames[i],'data/membrane/holdout/holdout/' + str(count) + '.params.pkl')
	    cv2.imwrite('data/membrane/holdout/holdout_label/' + str(count) + '.png', cv2.imread(y_holdout_filenames[i],0))
	    count += 1



