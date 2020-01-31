import numpy as np 
import os
import cv2
import pickle as pkl
from glob import glob
import sys

def stitch(tag):

	img_files = [img_file for img_file in glob('data/membrane/test/test/*.png') if 'predict' not in img_file]

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
			mask_file = 'data/membrane/test/test/' + file_base + '_predict.png'
			mask_tile = cv2.imread(mask_file,0)
			x = int(params['x']/2)
			y = int(params['y']/2)
			blank_mask[y+50:y+206,x+50:x+206] = mask_tile[50:206,50:206]
		cv2.imwrite('data/membrane/test_stitched/' + base + '.' + tag + '.stitched.png', blank_mask)


def holdout_stitch(tag):

	img_files = [img_file for img_file in glob('data/membrane/holdout/holdout/*.png') if 'predict' not in img_file]

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
			mask_file = 'data/membrane/holdout/holdout/' + file_base + '_predict.png'
			mask_tile = cv2.imread(mask_file,0)
			x = int(params['x']/2)
			y = int(params['y']/2)
			blank_mask[y+50:y+206,x+50:x+206] = mask_tile[50:206,50:206]
		cv2.imwrite('data/membrane/holdout_stitched/' + base + '.' + tag + '.stitched.png', blank_mask)


def main(tag):
	for i in range(1,11):
	    stitch(tag + '_epoch' + str(i))

"""
for terminal input
"""
if __name__ == '__main__':
    try:
        sys.argv[1]
    except:
        raise Exception('You need to pass in the tag.')
    basetag = str(sys.argv[1])

    main(basetag)




