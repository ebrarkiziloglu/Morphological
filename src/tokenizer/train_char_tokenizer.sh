#!/bin/bash
#SBATCH --partition a100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_train_char_tokenizer.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

echo "Starting pre-training at $(date)"
export HF_TOKEN=***REMOVED_HF_TOKEN***

python3 src/train_char_tokenizer.py

echo "Training completed at $(date)"
