print('starting')
from time import time
mt1 = time()
from model import *
from data import *
from stitch import *
from data_manager import *
from assess import *
import os
from glob import glob
import sys, contextlib
import random
from tensorflow import set_random_seed
import tensorflow as tf
import numpy as np
from time import time
from keras import backend as K
print('package loading finished')

"""
Printing suppression function.  Doesn't always work and no longer necessary.
"""
@contextlib.contextmanager
def suppress_print():
    save_stdout = sys.stdout
    sys.stdout = open('trash','w')
    yield
    sys.stdout = save_stdout

mt2 = time()
print('Time so far: ' + str(mt2-mt1) + ' s')

"""
Read in holdout set fold number from the command line.
Assign folds to holdout, test, and train sets.
"""
try:
	sys.argv[1]
	holdout_num = int(sys.argv[1])
except:
	holdout_num = 3
test_num = (holdout_num + 1)%5
train_num_a = (holdout_num + 2)%5
train_num_b = (holdout_num + 3)%5
train_num_c = (holdout_num + 4)%5

test_set = folds[test_num]
holdout_set = folds[holdout_num]
train_set = folds[train_num_a]|folds[train_num_b]|folds[train_num_c]

"""
Set random seeds in various packages
"""
random.seed(1)
np.random.seed(1)
set_random_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

"""
Variable n_mng sets the number of augmented 512x512 tiles
to produce for each MNG in the training set.
"""
n_mng = 500

"""
Generate tiles for the images in the test set.  
The U-Net needs a tile of a fixed 512x512 size, so the
test set images need to be turned into tiles.
Switchable.
"""
if True:
	print('Generating test tiles...')
	t1 = time()
	make_test_panels(test_set)
	t2 = time()
	print('Done in: ' + str(round(t2-t1,2)) + ' s')

#Reset random seeds
random.seed(1)
np.random.seed(1)

"""
Generate tiles for the images in the holdout set.  
The U-Net needs a tile of a fixed 512x512 size, so the
test set images need to be turned into tiles.
Switchable.
"""
if True:
	print('Generating holdout tiles...')
	t1 = time()
	make_holdout_panels(holdout_set)
	t2 = time()
	print('Done in: ' + str(round(t2-t1,2)) + ' s')

#Reset random seeds
random.seed(1)
np.random.seed(1)

"""
Generate 512x512 tiles for the training data.
For each MNG, the function generates an n_mng number
of augmented tiles.  
The augmentation has a random offset, placing the center 
of the MNG randomly on the tile, and it also loops through
the eight possible flip and rotation combinations.
The code begins by wiping any pre-existing directory of training tiles.
It also randomizes the order in which the files get fed in.
Switchable.
"""
if True:
	print('Generating training data...')
	t1 = time()
	n_files = get_training_data(train_set,n_mng=n_mng)
	t2 = time()
	print('Done in: ' + str(round(t2-t1,2)) + ' s')
else:
	n_files = len(glob('data/membrane/train/image/*.png'))

"""
Set random seeds in various packages
"""
random.seed(1)
np.random.seed(1)
set_random_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

"""
Set the number of epochs to run
"""
num_epochs = 15

"""
Hardcode the file paths for the test and holdout directories
"""
test_dir = 'data/membrane/test/test'
holdout_dir = 'data/membrane/holdout/holdout'

"""
Hardcode the batch sizes to iterate through
"""
batch_sizes = [3] # [32, 16, 8, 4, 2, 1]

"""
Run the model, looping through the batch sizes
"""
for batch_size in batch_sizes:

	print('Running model for batch size: ' + str(batch_size))
	t1 = time()


	#Make the generator to feed in the training data.  We use a generator for memory management.
	data_gen_args = dict() #Empty data generation arguments because we do the augmentation ourselves.
	myGene = trainGenerator(batch_size,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

	#Set the model names and base name for file management purposes
	model_base_name = 'fold' + str(holdout_num) + '_' + str(batch_size) + 'b' + str(n_mng) + 'n' + str(batch_size)
	model_name = 'n_search_' + model_base_name

	#Train the model and run it on the test set
	if False:
		#Loop through the epochs
		for i in list(range(1,num_epochs +1)):
			"""
			Train the model
			"""
			if True:
				#If it's the first epoch, load an untrained model.  Otherwise load the weights from the previous epoch.
				if i==1:
					model = unet()
				else:
					model = unet(pretrained_weights = model_name + '_epoch' + str(i-1) + '.hdf5')
				#Set the file name of output model and set some monitoring parameters
				model_checkpoint = ModelCheckpoint(model_name + '_epoch' + str(i) + '.hdf5', monitor='loss',verbose=1, save_best_only=True)
				#Run the model
				model.fit_generator(myGene,steps_per_epoch=int(float(n_files)/float(batch_size)),epochs=1,callbacks=[model_checkpoint])
			"""
			Apply the model to the test set
			"""
			#Calculate the number of tiles in the test set
			n = len(set(li.partition('.png')[0].partition('_')[0] for li in glob(test_dir + '/*.png')))
			print('\n\n***\n\nn:' + str(n))
			#Make a generator for the test set tiles
			testGene = testGenerator(test_dir, num_image = n)
			#Load the model for this epoch
			model = unet()
			model.load_weights(model_name + '_epoch' + str(i) + '.hdf5')
			#Make a generator for the prediction images
			results = model.predict_generator(testGene,n,verbose=1)
			print('Saving results (printing suppressed)...')
			st1 = time()
			with suppress_print():
				#Save the prediciton images (the test set tiles with the trained model applied)
				saveResult(test_dir,results)
			st2 = time()
			print('Done in: ' + str(round(st2-st1,2)) + ' s')

			print('Stitching...')
			st1 = time()
			#Stitch together the prediction tiles in the test set images to make test set prediciton images.
			#These images should have a prediction of where the MNGs are, based off the trained model.
			stitch(model_base_name + '_epoch' + str(i))
			st2 = time()
			print('Done in: ' + str(round(st2-st1,2)) + ' s')
	"""
	Run the grid search to optimize the three hyperparameters on the test set:
	1) epoch
	2) cutoff_brightness
	3) cutoff_area
	Save the tuned parameter dictionary to a pickle file.
	Switchable; if switch turned off, just load the tuned parameters from the saved pickle file.
	Results will be saved in the results directory.
	First, set random seeds in various packages.
	"""
	random.seed(1)
	np.random.seed(1)
	set_random_seed(1)
	os.environ['PYTHONHASHSEED']=str(1)
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)
	if True:
		tuned_params_dict = grid_search(model_base_name)
		pkl.dump(tuned_params_dict, open('tuned_params/tuned_params_dict.' + model_base_name + '.pkl','wb'))
	else:
		tuned_params_dict = pkl.load(open('tuned_params/tuned_params_dict.' + model_base_name + '.pkl','rb'))

	"""
	Using the tuned hyperparameters, apply the model to the holdout set.
	Switchable.
	"""
	if True:
		n = len(set(li.partition('.png')[0].partition('_')[0] for li in glob(holdout_dir + '/*.png')))
		print('\n\n***\n\nn:' + str(n))
		holdoutGene = testGenerator(holdout_dir, num_image = n)
		model = unet()
		model.load_weights(model_name + '_epoch' + str(int(tuned_params_dict['epoch'])) + '.hdf5')
		results = model.predict_generator(holdoutGene,n,verbose=1)
		print('Saving results (printing suppressed)...')
		st1 = time()
		with suppress_print():
			saveResult(holdout_dir,results)
		st2 = time()
		print('Done in: ' + str(round(st2-st1,2)) + ' s')

		print('Stitching...')
		st1 = time()
		holdout_stitch(model_base_name)
		st2 = time()
		print('Done in: ' + str(round(st2-st1,2)) + ' s')

	if True:
		print('Calculating holdout set f1 score...')
		st1 = time()
		calculate_holdout_f1(model_base_name, tuned_params_dict['cutoff_brightness'], \
			tuned_params_dict['cutoff_area'], tuned_params_dict['epoch'])
		st2 = time()
		print('Done in: ' + str(round(st2-st1,2)) + ' s')

	t2 = time()
	print('Full runtime for batch ' + str(batch_size) + ': ' + str(round(t2-t1,1)) + ' s')

mt2 = time()
print('Full runtime for all batches: ' + str(round(mt2-mt1,1)) + ' s')
