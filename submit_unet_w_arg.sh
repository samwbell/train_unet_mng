#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=95:59:00

# Default resources are 1 core with 2.8GB of memory

# Specify resources:
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -p gpu --gres=gpu:4

# Specify a job name:
#SBATCH -J full_data_run

# Specify an output file
#SBATCH -o full_data_run-%j.out
#SBATCH -e full_data_run-%j.out

# Run a command

module load tensorflow/1.5.0_gpu_py3
module load cuda/9.0.176 cudnn/7.0 python/3.5.2
module load keras/2.1.3_py3
module load scikit-image/0.15.0
module load h5py/2.9.0
module load opencv-python/4.1.0.25
module load scikit-learn/0.21.2
module load python/3.5.2
python3 main.py $1

