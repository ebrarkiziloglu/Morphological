#!/bin/bash
#SBATCH --partition preempt
#SBATCH --gres gpu:1
#SBATCH --output eval/logs/%j_eval_multiseed_char-zh-de_phase2_shifted.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
cd "${ROOT}"
DATIVE_JSON="${ROOT}/src/eval_dataset/dative_accusative_pairs_shifted.json"
ACCUSATIVE_JSON="${ROOT}/src/eval_dataset/accusative_dative_pairs_shifted.json"
JSON_PATHS="${DATIVE_JSON},${ACCUSATIVE_JSON}"
OUT_ROOT="${ROOT}/eval_results"
SEEDS="0,1,2,3,4"
EPOCHS="1,3,5"
PHASE_SUBDIR="phase_2/regular/de_nllb/shifted"
RUN_NAME="char-zh-de_${PHASE_SUBDIR//\//_}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${ROOT}/src/eval/logs"

TOKENIZER="${ROOT}/src/tokenizer/tokenizer_char/"

echo "=== Evaluating char-zh-de | ${PHASE_SUBDIR} | epochs=${EPOCHS} seeds=${SEEDS} | $(date) ==="
python3 ${ROOT}/src/eval/evaluate_ppl.py \
    --model_root "${ROOT}/models/v3/char-zh-de" \
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
