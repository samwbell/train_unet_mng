from model import *
from data import *
from stitch import stitch
from data_manager import *
from time import time
import os
from glob import glob
import sys, contextlib

@contextlib.contextmanager
def suppress_print():
    save_stdout = sys.stdout
    sys.stdout = open('trash','w')
    yield
    sys.stdout = save_stdout


# if not os.path.exists('data/membrane/holdout'):
# 	print('Generating holdout tiles...')
# 	t1 = time()
# 	make_holdout_panels()
# 	t2 = time()
# 	print('Done in: ' + str(round(t2-t1,2)) + ' s'

batch_size = 3
n_mng = 500
test_dir = 'data/membrane/test/test'

model_base_name = str(batch_size) + 'batches' + str(n_mng) + 'n' + str(batch_size)
model_name = 'full_data_' + model_base_name

n = len(set(li.partition('.png')[0].partition('_')[0] for li in glob(test_dir + '/*.png')))
print('\n\n***\n\nn:' + str(n))

for i in range(1,11):
	print(i)
	testGene = testGenerator(test_dir, num_image = n)
	model = unet()
	model.load_weights(model_name + '_epoch' + str(i) + '.hdf5')
	results = model.predict_generator(testGene,n,verbose=1)
	print('Saving results (printing suppressed)...')
	st1 = time()
	with suppress_print():
		saveResult(test_dir,results)
	st2 = time()
	print('Done in: ' + str(round(st2-st1,2)) + ' s')

	print('Stitching...')
	st1 = time()
	stitch(model_base_name + '_epoch' + str(i))
	st2 = time()
	print('Done in: ' + str(round(st2-st1,2)) + ' s')

