#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_de_only_shifted_phase2_nllb_seed_0.out
#SBATCH --time 24:00:00
SEEDS=(0)

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
# Layout under each language output dir:
#   seed_<S>/Phase1_epoch_5
#   seed_<S>/Phase1_epoch_10
#   seed_<S>/phase1_<l1>              # alias of final Phase-1 state
#   seed_<S>/phase2_from_ep10/Phase2_epoch_{1..5}
#   seed_<S>/phase2_from_ep10/phase2_<l1>_deu

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
        --batch_size 16 \
        --epochs_phase2 5 \
        --lr_phase2 5e-5 \
        --grad_acc 2 \
        --logging_steps 100 \
        --seed "${SEED}"
done

echo "Finished multi-seed pre-training at $(date)"
