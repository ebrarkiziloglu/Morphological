#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_gpt-reset-emb-tr-lr2-5e-5.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

echo "Starting pre-training at $(date)"
 
# python3 src/train.py \
#     --tokenizer_path ~/Morphological/src/tokenizer/tokenizer_char/ \
#     --resume_from ~/Morphological/models/v3/char-en-de/phase1_eng \
#     --phase2_path ~/Morphological/data/BabyLM_DE_subset/BabyLM_DE_subset.de \
#     --l1 eng \
#     --l2 deu \
#     --out ~/Morphological/models/v3/gpt-reset-emb-en-de-v2/ \
#     --max_seq_len 1024 \
#     --max_position_embeddings 1024 \
#     --batch_size 16 \
#     --epochs_phase2 5 \
#     --lr 1e-4 \
#     --lr_phase2 5e-5 \
#     --grad_acc 2 \
#     --logging_steps 100 \
#     --seed 42 \
#     --reset_embeddings


python3 src/train.py \
    --tokenizer_path ~/Morphological/src/tokenizer/tokenizer_char/ \
    --resume_from ~/Morphological/models/v3/char-tr-de/phase1_tur \
    --phase2_path ~/Morphological/data/BabyLM_DE_subset/BabyLM_DE_subset.de \
    --l1 tur \
    --l2 deu \
    --out ~/Morphological/models/v3/reset-emb-tr-de/ \
    --max_seq_len 1024 \
    --max_position_embeddings 1024 \
    --batch_size 16 \
    --epochs_phase2 5 \
    --lr 1e-4 \
    --lr_phase2 5e-5 \
    --grad_acc 2 \
    --logging_steps 100 \
    --seed 42 \
    --reset_embeddings
