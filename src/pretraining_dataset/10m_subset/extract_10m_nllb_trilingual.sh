#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_extract_10m_nllb_trilingual.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

python pretraining_dataset/extract_10m_nllb_trilingual.py