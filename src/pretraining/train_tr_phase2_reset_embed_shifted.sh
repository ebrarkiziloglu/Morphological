#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_tr_phase2_reset_embed_shifted_nllb_seed_4.out
#SBATCH --time 24:00:00
SEEDS=(4)

# Multi-seed two-phase pretraining
# --------------------------------
# Phase 1 (EN and TR separately):
#   - LR = 1e-4, 10 epochs
#   - checkpoints after epochs 5 and 10
#   - 5 seeds
# Phase 2 (German, resume from Phase-1 epoch-10 of the same seed):
#   - LR = 5e-5, 5 epochs
#   - checkpoint after every epoch
#   - 5 seeds
#
# Checkpoint layout under --out:
#   ep{E}/seed_{S}/
# e.g. phase2_from_ep10/ep1/seed_0 ... phase2_from_ep10/ep5/seed_0
#
# Phase-1 resume still uses the legacy path seed_<S>/Phase1_epoch_10
# if those checkpoints were trained with the old layout.

source ~/.bashrc
conda activate morph2

echo "Starting multi-seed pre-training at $(date)"
export HF_TOKEN=***REMOVED_HF_TOKEN***

ROOT="${HOME}/Morphological"
TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"
# PHASE2_DATA="${ROOT}/data/BabyLM_DE_subset/BabyLM_DE_subset_shifted.de"
PHASE2_DATA="${ROOT}/data/NLLB_10M_subset/NLLB_10M_shifted.de"

# Five independent training seeds
# SEEDS=(0 1 2 3 4)
TR_OUT="${ROOT}/models/v3/char-tr-de"
TR_OUT_PHASE2="${TR_OUT}/phase_2/regular/de_nllb/reset_embeddings_shifted"
# ---------------------------------------------------------------------------
# Phase 2 — Shifted German from TR Phase-1 epoch 5 (same seed)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    CKPT="${TR_OUT}/phase_1/regular/ep5/seed_${SEED}/"
    echo "=== Phase 2 DE from TR ep5 | seed=${SEED} | ckpt=${CKPT} | $(date) ==="
    python3 pretraining/train.py \
        --resume_from "${CKPT}" \
        --phase2_path "${PHASE2_DATA}" \
        --l1 tur \
        --l2 deu \
        --out "${TR_OUT_PHASE2}/" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
        --batch_size 16 \
        --epochs_phase2 5 \
        --save_epochs_phase2 "1,2,3,4,5" \
        --lr_phase2 5e-5 \
        --grad_acc 2 \
        --logging_steps 100 \
        --seed "${SEED}" \
        --reset_embeddings
done

echo "Finished multi-seed pre-training at $(date)"
