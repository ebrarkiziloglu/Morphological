#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output %j.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

echo "Starting German extraction at $(date)"
# Pass your token as an argument or ensure it's in your environment
export HF_TOKEN=***REMOVED_HF_TOKEN***
python3 src/extract_de_babylm.py --token $HF_TOKEN \
    --analyze-only \
    --original-bytes 61_702_345 \
    --original-mean 218.73 \
    --original-median 193.00 \
    --original-std 101.86 \
    --byte-premium 1.0537171

echo "Extraction completed at $(date)"