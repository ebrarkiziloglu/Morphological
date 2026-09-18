#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output %j.out
#SBATCH --time 24:00:00

# Load conda
source ~/.bashrc

# Activate conda environment
conda activate babylm-pretrain

conda install cmake

cd ~/sfst/build
cmake ..
make

export PATH=~/sfst/build/src:$PATH