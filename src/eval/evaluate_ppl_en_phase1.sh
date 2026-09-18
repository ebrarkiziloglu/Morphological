#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_EVAL_MULTISEED_char-en-de_phase1_shuffled.out
#SBATCH --time 24:00:00

# Multi-seed / multi-epoch PPL preference evaluation
# -------------------------------------------------
# Checkpoint layout (v3 char-en-de):
#   {model_root}/{phase_subdir}/ep{k}/seed_{s}/   for k=1..5, s=0..4
#   {model_root}/{phase_subdir}/ep0/              seedless baseline
# Tokenizer: loaded once from ep0 and reused for all checkpoints.
#
# CSVs:
#   *_per_seed_overall.csv
#   *_aggregate_overall.csv          <-- mean over seeds (use this)
#   *_per_seed_by_category.csv
#   *_aggregate_by_category.csv

source ~/.bashrc
conda activate morph2

echo "Starting multi-seed PPL evaluation at $(date)"
 
ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
cd "${ROOT}"
DATIVE_JSON="${ROOT}/src/eval_dataset/dative_accusative_pairs.json"
ACCUSATIVE_JSON="${ROOT}/src/eval_dataset/accusative_dative_pairs.json"
JSON_PATHS="${DATIVE_JSON},${ACCUSATIVE_JSON}"
OUT_ROOT="${ROOT}/results"
SEEDS="0,1,2,3,4"
EPOCHS="5"
PHASE_SUBDIR="phase_1/shuffled"
RUN_NAME="char-en-de_${PHASE_SUBDIR//\//_}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${ROOT}/src/eval/logs"

TOKENIZER_PATH="${ROOT}/models/v3/char-en-de/"

echo "=== Evaluating char-en-de | ${PHASE_SUBDIR} | epochs=${EPOCHS} seeds=${SEEDS} | $(date) ==="
python3 ${ROOT}/src/eval/evaluate_ppl.py \
    --model_root "${ROOT}/models/v3/char-en-de" \
    --tokenizer_path "${TOKENIZER_PATH}" \
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

echo "Finished multi-seed PPL evaluation at $(date)"
echo "Aggregate CSVs (seed averages):"
echo "  ${OUT_ROOT}/${RUN_NAME}/"
