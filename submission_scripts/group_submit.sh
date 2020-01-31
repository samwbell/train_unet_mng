#!/bin/bash

FIRST=$(sbatch submit_unet_w_arg.sh 1 | cut -f 4 -d' ')
echo $FIRST
SECOND=$(sbatch -d afterany:$FIRST submit_unet_w_arg.sh 2 | cut -f 4 -d' ')
echo $SECOND
THIRD=$(sbatch -d afterany:$SECOND submit_unet_w_arg.sh 3 | cut -f 4 -d' ')
echo $THIRD
FOURTH=$(sbatch -d afterany:$THIRD submit_unet_w_arg.sh 4 | cut -f 4 -d' ')
echo $FOURTH
FIFTH=$(sbatch -d afterany:$FOURTH submit_unet_w_arg.sh 0 | cut -f 4 -d' ')
echo $FIFTH

exit 0
