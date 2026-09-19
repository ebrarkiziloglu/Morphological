#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_tr_stripped_phase2_de_nllb_regular_seeds_0_1_2_3.out
#SBATCH --time 24:00:00
SEEDS=(0 1 2 3)

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
#
# Checkpoint layout under --out:
#   ep{E}/seed_{S}/
# e.g. phase2_from_ep10/ep1/seed_0 ... phase2_from_ep10/ep5/seed_0

source ~/.bashrc
conda activate morph2

echo "Starting multi-seed pre-training at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"
PHASE2_DATA="${ROOT}/data/NLLB_10M_subset/NLLB_10M.de"

TR_OUT="${ROOT}/models/v3/char-tr-de"

# From REGULAR Phase 1:
TR_OUT_PHASE2="${TR_OUT}/phase_2/stripped/regular/de_nllb/regular"

# From SHUFFLED Phase 1:
# TR_OUT_PHASE2="${TR_OUT}/phase_2/stripped/shuffled/de_nllb/regular"

# ---------------------------------------------------------------------------
# Phase 2 — German from TR Phase-1 epoch 5 (same seed)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    CKPT="${TR_OUT}/phase_1/stripped/regular/ep5/seed_${SEED}/"
    # CKPT="${TR_OUT}/phase_1/stripped/shuffled/ep5/seed_${SEED}/"
    echo "=== Phase 2 DE from TR ep5 | seed=${SEED} | ckpt=${CKPT} | $(date) ==="
    python3 pretraining/train.py \
        --resume_from "${CKPT}" \
        --phase2_path "${PHASE2_DATA}" \
        --tokenizer_path "${TOKENIZER}" \
        --l1 tur \
        --l2 deu \
        --out "${TR_OUT_PHASE2}/" \
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
