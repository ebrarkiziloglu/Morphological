#!/bin/bash
#SBATCH --partition preempt
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_shuffle-NLLB-TR-stripped-words-pretraining-dataset.out
#SBATCH --time 24:00:00

source ~/.bashrc
conda activate morph2

set -euo pipefail

IN_DIR=/hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_10M_subset
OUT_DIR=/hnvme/workspace/b279bb15-morph/Morphological/data/NLLB_10M_subset_shuffled

DIST=geometric
RATE=0.3
GEOM_P=0.5
MAX_D=20
SEED=42

mkdir -p "${OUT_DIR}"

python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/shuffle_words.py \
    --input  "${IN_DIR}/NLLB_10M_stripped.tr" \
    --output_dir "${OUT_DIR}" \
    --tokenize       "whitespace" \
    --corruption_rate "${RATE}" \
    --distribution   "${DIST}" \
    --max_distance   "${MAX_D}" \
    --geom_p         "${GEOM_P}" \
    --seed           "${SEED}" \
    --save_config


python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/shuffle_words.py \
    --input  "${IN_DIR}/NLLB_10M_stripped_nominal_infl.tr" \
    --output_dir "${OUT_DIR}" \
    --tokenize       "whitespace" \
    --corruption_rate "${RATE}" \
    --distribution   "${DIST}" \
    --max_distance   "${MAX_D}" \
    --geom_p         "${GEOM_P}" \
    --seed           "${SEED}" \
    --save_config

# for LANG in en tr zh; do
#     # NLLB ZH is largely unsegmented; character tokens are required so
#     # displacement actually moves units. EN/TR stay whitespace-tokenized.
#     if [[ "${LANG}" == "zh" ]]; then
#         TOKENIZE=chars
#     else
#         TOKENIZE=whitespace
#     fi

#     echo "=== Shuffling ${LANG} (tokenize=${TOKENIZE}) ==="
#     python3 /hnvme/workspace/b279bb15-morph/Morphological/src/pretraining_dataset/shuffle_words.py \
#         --input  "${IN_DIR}/NLLB_10M.${LANG}" \
#         --output_dir "${OUT_DIR}" \
#         --tokenize       "${TOKENIZE}" \
#         --corruption_rate "${RATE}" \
#         --distribution   "${DIST}" \
#         --max_distance   "${MAX_D}" \
#         --geom_p         "${GEOM_P}" \
#         --seed           "${SEED}" \
#         --save_config
# done

# echo "Done at $(date)"
