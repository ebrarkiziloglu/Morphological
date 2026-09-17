#!/bin/bash
#SBATCH --partition a40
#SBATCH --gres gpu:1
#SBATCH --output logs/%j_EVAL_PPL_HF_DE.out
#SBATCH --time 24:00:00

# Evaluate PPL of gold vs corrupted sentences for Hugging Face checkpoints.
# Testsets default from whether --subdir contains 'shifted'.

source ~/.bashrc
conda activate morph2

echo "Starting multi-seed PPL evaluation at $(date)"
export HF_TOKEN=***REMOVED_HF_TOKEN***
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export HF_HOME=/home/vault/b279bb/b279bb15/hf_home
mkdir -p "${HF_HOME}"

ROOT="${HOME}/Morphological"
cd "${ROOT}"
REPO_IDS="morph-char-de"
SUBDIR="phase_2/de_nllb/regular,phase_2/de_nllb/shifted" #,phase_2/de_babylm/regular,phase_2/de_babylm/shifted"

STAGE_DIR="/home/vault/b279bb/b279bb15/hf_stage"

python3 ${ROOT}/src/eval/evaluate_ppl_hf.py \
    --repo_id "${REPO_IDS}" \
    --subdir "${SUBDIR}" \
    --stage_dir "${STAGE_DIR}" \
    --local_only
