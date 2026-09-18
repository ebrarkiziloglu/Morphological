#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output %j.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

echo "Starting German extraction at $(date)"
# HF_TOKEN is expected to be exported from ~/.bashrc

python3 pretraining_dataset/10m_subset/extract_10m_nllb_german.py --token $HF_TOKEN \
    --original-bytes 59_652_031 \
    --original-mean 125.25 \
    --original-median 114.00 \
    --original-std 52.08 \
    --byte-premium 1.0537171

echo "Extraction completed at $(date)"