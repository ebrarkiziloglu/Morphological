#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_shift_chars.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

echo "Starting pre-training at $(date)"

# Clean old outputs

# Re-run with fixed script
conda activate morph2 && python3 ~/Morphological/src/pretraining_dataset/shift_chars.py \
  --pretrain_path "../data/NLLB_10M_subset/NLLB_10M.de"