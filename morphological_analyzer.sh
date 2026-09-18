#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output %j.out
#SBATCH --time 24:00:00

# Load conda
source ~/.bashrc

# Activate conda environment
conda activate morph2
export PATH=~/sfst/build/src:$PATH

# python3 morphological_analyzer.py \
#   --tr-path "./data/arrow/OpenSubtitles_2024_en_tr/" \
#   --dataset_name "OpenSubtitles_2024_en_tr" \
#   --max-samples 1000000 \
#   --arrow True \
#   --morph-path "morphological_analyzer/TRMOR/morph.ca" \
#   --fst-infl2-path "morphological_analyzer/SMOR/bin/fst-infl2"

# python3 morphological_analyzer.py \
#   --tr-path "./data/arrow/HPLT_en_tr/" \
#   --dataset_name "HPLT_en_tr" \
#   --max-samples 1000000 \
#   --arrow True \
#   --morph-path "morphological_analyzer/TRMOR/morph.ca" \
#   --fst-infl2-path "morphological_analyzer/SMOR/bin/fst-infl2"

python3 morphological_analyzer.py \
  --tr-path "./data/arrow/split/OpenSubtitles_tr/" \
  --dataset_name "OpenSubtitles_tr_1M_with_sentences" \
  --max-samples 1000000 \
  --arrow True \
  --morph-path "morphological_analyzer/TRMOR/morph.ca" \
  --fst-infl2-path "morphological_analyzer/SMOR/bin/fst-infl2"

python3 morphological_analyzer.py \
  --tr-path "./data/arrow/split/HPLT_tr/" \
  --dataset_name "HPLT_tr_1M_with_sentences" \
  --max-samples 1000000 \
  --arrow True \
  --morph-path "morphological_analyzer/TRMOR/morph.ca" \
  --fst-infl2-path "morphological_analyzer/SMOR/bin/fst-infl2"
