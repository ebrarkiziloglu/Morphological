#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_remove_cases_from_tr_nllb.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

# Compute nodes cannot reach Hugging Face Xet CAS; use HTTPS + the login-node cache.
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=***REMOVED_HF_TOKEN***

python3 pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py \
   --input /home/hpc/b279bb/b279bb15/Morphological/data/NLLB_10M_subset/NLLB_10M.tr \
   --output /home/hpc/b279bb/b279bb15/Morphological/data/NLLB_10M_subset/NLLB_10M_stripped.tr