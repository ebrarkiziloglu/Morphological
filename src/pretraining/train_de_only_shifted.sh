#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_de_only_phase2_shifted_seeds_4.out
#SBATCH --time 24:00:00
SEEDS=(4)

source ~/.bashrc
conda activate morph2

echo "Starting multi-seed pre-training at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"
PHASE2_DATA="${ROOT}/data/NLLB_10M_subset/NLLB_10M_shifted.de"

DE_OUT="${ROOT}/models/v3/char-de/phase_2/de_nllb/shifted"

for SEED in "${SEEDS[@]}"; do
    echo "=== Phase 2 DE from scratch | seed=${SEED} | $(date) ==="
    python3 pretraining/train.py \
        --phase1_path "${PHASE2_DATA}" \
        --phase2_path "${PHASE2_DATA}" \
        --epochs_phase1 0 \
        --tokenizer_path "${TOKENIZER}" \
        --l1 deu \
        --l2 deu \
        --out "${DE_OUT}/" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
         --batch_size 32 \
        --epochs_phase2 5 \
        --lr_phase2 5e-5 \
        --grad_acc 2 \
        --logging_steps 100 \
        --seed "${SEED}" \
        --attn_implementation flash_attention_3 \
        --bf16
done

echo "Finished multi-seed pre-training at $(date)"
