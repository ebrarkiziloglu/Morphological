#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_tr_stripped_phase1_regular_or_shuffled_seeds_0.out
#SBATCH --time 24:00:00
SEEDS=(0)

# Multi-seed two-phase pretraining
# --------------------------------
# Phase 1 (EN, TR and ZH separately):
#   - LR = 1e-4, 5 epochs
#   - checkpoints after epochs 5
#   - 5 seeds
# Phase 2 (German, resume from Phase-1 epoch-5 of the same seed):
#   - LR = 5e-5, 5 epochs
#   - checkpoint after every epoch
#   - 5 seeds

source ~/.bashrc
conda activate morph2

echo "Starting multi-seed pre-training at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"

# ---------------------------------------------------------------------------
# Phase 1 — Turkish (NLLB 10M)
# ---------------------------------------------------------------------------

# REGULAR:
TR_P1="${ROOT}/data/NLLB_10M_subset/NLLB_10M_stripped.tr"
TR_OUT="${ROOT}/models/v3/char-tr-de/phase_1/stripped/regular/"

# SHUFFLED:
# TR_P1="${ROOT}/data/NLLB_10M_subset_shuffled/NLLB_10M_stripped_shuffled_geometric_r0.3.tr"
# TR_OUT="${ROOT}/models/v3/char-tr-de/phase_1/stripped/shuffled/"

for SEED in "${SEEDS[@]}"; do
    echo "=== Phase 1 Turkish | seed=${SEED} | $(date) ==="
    python3 pretraining/train.py \
        --tokenizer_path "${TOKENIZER}" \
        --phase1_path "${TR_P1}" \
        --l1 tur \
        --l2 deu \
        --out "${TR_OUT}" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
        --batch_size 16 \
        --epochs_phase1 5 \
        --epochs_phase2 0 \
        --save_epochs_phase1 5 \
        --lr 1e-4 \
        --grad_acc 2 \
        --logging_steps 100 \
        --seed "${SEED}" \
        --attn_implementation flash_attention_3 \
        --bf16
done

echo "Finished multi-seed pre-training at $(date)"
