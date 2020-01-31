from model import *
from data import *
import os, shutil
from glob import glob
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

"""
Writes a string to a file
"""
def write_file(path, text):
    with open(path, 'w') as path_stream:
        path_stream.write(text)
        path_stream.close()

def make_image_panels(image_file):
    random.seed(1)
    if os.path.exists('prediction/panels'):
        shutil.rmtree('prediction/panels')
    os.mkdir('prediction/panels')

    base = image_file.rpartition('.')[0].rpartition('/')[2]
    #print(base)
    imx = cv2.imread(image_file,0)
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
            cv2.imwrite('prediction/panels/' + str(i) + '.png',imx[top:bottom,left:right])
            params = {'x':x,'y':y,'i':i,'base':base}
            pkl.dump(params,open('prediction/panels/' + str(i) + '.params.pkl', 'wb'))
            i += 1

def predict():
	n = len(set(li.partition('.png')[0].partition('_')[0] for li in glob('prediction/panels/*.png')))
	print('\n\n***\n\nn:' + str(n))
	testGene = testGenerator("prediction/panels", num_image = n)
	model = unet()
	model.load_weights("unet_mng_model.hdf5")
	results = model.predict_generator(testGene,n,verbose=1)
	saveResult("prediction/panels",results)

def stitch():
    img_files = [img_file for img_file in glob('prediction/panels/*.png') if '_predict.png' not in img_file]

    print(img_files)
    base_dict = {}

    params_list = []
    for img_file in img_files:
    	param_file = img_file.rpartition('.png')[0] + '.params.pkl'
    	params = pkl.load(open(param_file, 'rb'))
    	params['file'] = img_file
    	params_list.append(params)

    base_set = set(params['base'] for params in params_list)

    for base in base_set:
    	base_dict[base] = []

    for params in params_list:
    	base_dict[params['base']].append(params)

    for base in base_set:
    	print(base)
    	xmax = max([params['x'] for params in base_dict[base]]) + 512
    	ymax = max([params['y'] for params in base_dict[base]]) + 512
    	blank_mask = np.zeros((int(ymax/2),int(xmax/2)), np.uint8)
    	for params in base_dict[base]:
    		file_base = params['file'].rpartition('/')[2].rpartition('.png')[0]
    		mask_file = 'prediction/panels/' + file_base + '_predict.png'
    		mask_tile = cv2.imread(mask_file,0)
    		x = int(params['x']/2)
    		y = int(params['y']/2)
    		blank_mask[y+50:y+206,x+50:x+206] = mask_tile[50:206,50:206]
    	cv2.imwrite('prediction/stitched/' + base + '.stitched.png', blank_mask)

def apply_cutoffs(image_file):
    param_dict = pkl.load(open('unet_mng_model.params.pkl', 'rb'))
    cutoff_brightness = param_dict['cutoff_brightness']
    cutoff_area = param_dict['cutoff_area']
    base = image_file.rpartition('.')[0].rpartition('/')[2]
    prediction_file = 'prediction/stitched/' + base + '.stitched.png'
    prediction = cv2.imread(prediction_file)

    base = prediction_file.rpartition('/')[2].rpartition('.png')[0]

    pthresh = cv2.threshold(prediction, cutoff_brightness, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('prediction/stitched/' + base + '_binary.png', pthresh)
    pthresh = cv2.imread('prediction/stitched/' + base + '_binary.png',0)
    h,w = pthresh.shape[:2]

    basebase = base.rpartition('.')[0].rpartition('.')[0]

    filtered = cv2.imread('prediction/stitched/' + base + '_binary.png')
    cv2.rectangle(filtered,(0,0),(w,h), (255,255,255),-1)


    filtered_bw = pthresh.copy()

    count = 0
    contours, im2  = cv2.findContours(filtered_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.9*h*w and area > 10:
            M = cv2.moments(cnt)
            #cv2.drawContours(filtered,[cnt],0,(0,0,0),2)
            if M['m00'] != 0 and area > cutoff_area:
                cv2.drawContours(filtered,[cnt],0,(0,0,0),-1)
                count += 1


    cv2.imwrite('prediction/predictions/' + base + '.mngs.png', filtered)
    write_file('prediction/predictions/' + base + '.mng_count.txt', str(count))

    return count

random.seed(1)
np.random.seed(1)
set_random_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

image_files = glob('prediction/images/*')
tt1 = time.time()
for image_file in image_files:
    mt1 = time.time()

    print('Making panels...')
    t1 = time.time()
    make_image_panels(image_file)
    t2 = time.time()
    print('Run time: ' + str(round(t2-t1, 2)) + ' s')

    print('Predicting...')
    t1 = time.time()
    predict()
    t2 = time.time()
    print('Run time: ' + str(round(t2-t1, 2)) + ' s')

    print('Stitching...')
    t1 = time.time()
    stitch()
    t2 = time.time()
    print('Run time: ' + str(round(t2-t1, 2)) + ' s')

    print('Applying cutoffs...')
    t1 = time.time()
    apply_cutoffs(image_file)
    t2 = time.time()
    print('Run time: ' + str(round(t2-t1, 2)) + ' s')

    mt2 = time.time()
    print('Full run time for ' + image_file.rpartition('/')[2] + ': ' + str(round(mt2-mt1, 2)) + ' s')
tt2=time.time()
print('Total run time for all ' + str(len(image_files)) + ' images: ' + str(round(tt2-tt1, 2)) + ' s')


