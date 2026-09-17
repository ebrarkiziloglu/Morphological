#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output %j.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

mkdir -p logs

echo "Starting extraction at $(date)"
python3 src/extract_10m_hplt.py
echo "Extraction completed at $(date)"