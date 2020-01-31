# train_unet_mng

This code is designed to train a U-Net convolutional neural network.  Much of this code is modified from: https://github.com/zhixuhao/unet

This code is written for installation on a particular cluster, and the module loading and batch scripts are not applicable to other systems.  These scripts can be found within the submission_scripts folder.  The processing_scripts contains saved versions of scripts used for data preprocessing, as well as saved older versions of parts of the code.

To run the code, run main.py with a command line argument for the holdout set number ranging from 0 to 4.

