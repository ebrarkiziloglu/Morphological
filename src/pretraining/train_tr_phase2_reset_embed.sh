#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_tr_phase2_reset_embed_train_eval_all_seeds.out
#SBATCH --time 24:00:00
SEEDS=(0 1 2 3 4)

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
TR_OUT_PHASE2="${TR_OUT}/phase_2/regular/de_nllb/reset_embeddings"

# ---------------------------------------------------------------------------
# Phase 2 — German from TR Phase-1 epoch 5 (same seed)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    CKPT="${TR_OUT}/phase_1/regular/ep5/seed_${SEED}/"
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
        --bf16 \
        --reset_embeddings
done

echo "Finished multi-seed pre-training at $(date)"

cd "${ROOT}"
DATIVE_JSON="${ROOT}/src/eval_dataset/dative_accusative_pairs.json"
ACCUSATIVE_JSON="${ROOT}/src/eval_dataset/accusative_dative_pairs.json"
JSON_PATHS="${DATIVE_JSON},${ACCUSATIVE_JSON}"
OUT_ROOT="${ROOT}/eval_results"
SEEDS="0,1,2,3,4"
EPOCHS="1,3,5"
PHASE_SUBDIR="phase_2/regular/de_nllb/reset_embeddings"
RUN_NAME="char-tr-de_${PHASE_SUBDIR//\//_}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${ROOT}/src/eval/logs"

TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"

echo "=== Evaluating char-tr-de | ${PHASE_SUBDIR} | epochs=${EPOCHS} seeds=${SEEDS} | $(date) ==="
python3 ${ROOT}/src/eval/evaluate_ppl.py \
    --model_root "${ROOT}/models/v3/char-tr-de" \
    --tokenizer_path "${TOKENIZER}" \
    --run_name "${RUN_NAME}" \
    --phase_subdir "${PHASE_SUBDIR}" \
    --seeds "${SEEDS}" \
    --epochs "${EPOCHS}" \
    --json_path "${JSON_PATHS}" \
    --gold_key gold_text \
    --corrupt_key corrupted \
    --category_key category \
    --max_length 1024 \
    --out_dir "${OUT_ROOT}/${RUN_NAME}" \
    --bootstrap_n 0

echo "Aggregate CSVs (seed averages):"
echo "  ${OUT_ROOT}/${RUN_NAME}/"
