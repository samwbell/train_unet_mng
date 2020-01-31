from skimage import data, img_as_float
from skimage import exposure
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


color = cv2.imread('cropped/30159.1.png')
grayscale = cv2.imread('cropped/30159.1.png',0)

#cv2.imwrite('cropped/30159.1.gray.png', color)

print(color.mean())
print(grayscale.mean())

cf = 167.4220504227964/color.mean()
cf = 170.4220504227964/color.mean()
rescaled = (color.astype(np.float)*cf).astype(np.uint8)

cv2.imwrite('cropped/30159.1.rescaled.jpg', rescaled)


