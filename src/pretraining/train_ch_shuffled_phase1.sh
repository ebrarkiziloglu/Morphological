#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_ch_shuffled_phase1_seed_4.out
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
PHASE2_DATA="${ROOT}/data/BabyLM_DE_subset/BabyLM_DE_subset.de"

# Five independent training seeds
# ---------------------------------------------------------------------------
# Phase 1 — Chinese (NLLB 10M) Shuffled
# ---------------------------------------------------------------------------
CH_P1="${ROOT}/data/NLLB_10M_subset_shuffled/NLLB_10M_shuffled_geometric_r0.3.zh"
CH_OUT="${ROOT}/models/v3/char-zh-de/phase_1/shuffled/"

for SEED in "${SEEDS[@]}"; do
    echo "=== Phase 1 Chinese | seed=${SEED} | $(date) ==="
    python3 pretraining/train.py \
        --tokenizer_path "${TOKENIZER}" \
        --phase1_path "${CH_P1}" \
        --l1 zh \
        --l2 deu \
        --out "${CH_OUT}" \
        --max_seq_len 1024 \
        --max_position_embeddings 1024 \
        --batch_size 16 \
        --epochs_phase1 5 \
        --epochs_phase2 0 \
        --save_epochs_phase1 5 \
        --lr 1e-4 \
        --grad_acc 2 \
        --logging_steps 100 \
        --seed "${SEED}"
done

echo "Finished multi-seed pre-training at $(date)"
