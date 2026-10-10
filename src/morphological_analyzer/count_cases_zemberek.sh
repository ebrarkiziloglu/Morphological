#!/bin/bash
#SBATCH --partition preempt
#SBATCH --gres gpu:1
#SBATCH --output logs/count_cases_zemberek_postp.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

# python morphological_analyzer/count_cases_zemberek.py \
#     --tr-path ../data/NLLB_10M_subset/NLLB_10M.tr \
#     --dataset_name NLLB_10M \
#     --num-workers 32

# python morphological_analyzer/count_cases_zemberek.py \
#     --tr-path ../data/NLLB_10M_subset/NLLB_10M_stripped.tr \
#     --dataset_name NLLB_10M_stripped \
#     --num-workers 32

# python morphological_analyzer/count_cases_zemberek.py \
#     --tr-path ../data/NLLB_10M_subset/NLLB_10M_stripped_nominal.tr \
#     --dataset_name NLLB_10M_stripped_nominal \
#     --num-workers 32

python morphological_analyzer/count_cases_zemberek.py \
    --tr-path ../data/NLLB_10M_subset/NLLB_10M_stripped_postp.tr \
    --dataset_name NLLB_10M_stripped_postp \
    --num-workers 32  