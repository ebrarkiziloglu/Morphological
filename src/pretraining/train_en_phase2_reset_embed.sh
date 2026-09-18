#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_en_phase2_reset_embed_seeds_0.out
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
#
# Checkpoint layout under --out:
#   ep{E}/seed_{S}/
# e.g. phase2_from_ep10/ep1/seed_0 ... phase2_from_ep10/ep5/seed_0

source ~/.bashrc
conda activate morph2

# Flash Attention 3 (H100/Hopper only, --attn_implementation flash_attention_3
# below). morph2 has torch==2.6.0+cu124 and `kernels` installed, so
# transformers auto-downloads a prebuilt FA3 kernel from the HF Hub the first
# time it's needed (no flash_attn_3 build required) and caches it under
# ~/.cache/huggingface. The compute node needs outbound network access on
# first run; after that it's cached.

echo "Starting multi-seed pre-training at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"
PHASE2_DATA="${ROOT}/data/NLLB_10M_subset/NLLB_10M.de"

EN_OUT="${ROOT}/models/v3/char-en-de"

# From REGULAR Phase 1:
EN_OUT_PHASE2="${EN_OUT}/phase_2/regular/de_nllb/reset_embeddings"

# ---------------------------------------------------------------------------
# Phase 2 — German from EN Phase-1 epoch 5 (same seed)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    CKPT="${EN_OUT}/phase_1/regular/ep5/seed_${SEED}/"
    echo "=== Phase 2 DE from EN ep5 | seed=${SEED} | ckpt=${CKPT} | $(date) ==="
    python3 pretraining/train.py \
        --resume_from "${CKPT}" \
        --phase2_path "${PHASE2_DATA}" \
        --tokenizer_path "${TOKENIZER}" \
        --l1 eng \
        --l2 deu \
        --out "${EN_OUT_PHASE2}/" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
        --batch_size 16 \
        --epochs_phase2 5 \
        --lr_phase2 5e-5 \
        --grad_acc 2 \
        --logging_steps 100 \
        --seed "${SEED}" \
        --attn_implementation flash_attention_3 \
        --bf16 \
        --reset_embeddings
done

echo "Finished multi-seed pre-training at $(date)"
