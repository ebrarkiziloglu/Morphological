#!/bin/bash
# Run on a login node (not a GPU compute node). Compute nodes time out
# connecting to the Hugging Face weight CDN.

source ~/.bashrc
conda activate morph2

echo "Starting HF checkpoint download at $(date)"
 export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export HF_HOME=/home/vault/b279bb/b279bb15/hf_home
STAGE_DIR="/home/vault/b279bb/b279bb15/hf_stage"
mkdir -p "${HF_HOME}" "${STAGE_DIR}"

ROOT="/hnvme/workspace/b279bb15-morph/Morphological"
cd "${ROOT}"
REPO_IDS="morph-char-zh-de"
SUBDIR="phase_1/regular,phase_1/shuffled,phase_2/regular/de_nllb/regular,phase_2/regular/de_nllb/reset_embeddings,phase_2/shuffled/de_nllb/regular"

python3 ${ROOT}/src/eval/evaluate_ppl_hf.py \
    --repo_id "${REPO_IDS}" \
    --subdir "${SUBDIR}" \
    --stage_dir "${STAGE_DIR}" \
    --download_only

echo "Finished HF checkpoint download at $(date)"
