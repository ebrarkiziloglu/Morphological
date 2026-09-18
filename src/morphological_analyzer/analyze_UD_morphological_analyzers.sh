#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --output %j.out
#SBATCH --time 24:00:00

# Load conda
source ~/.bashrc

# Activate conda environment
conda activate morph2

# SMOR:
python analyze_UD.py --analyzer smor --max-sentences 20000

# SPACY:
python analyze_UD.py --analyzer spacy_de --ud-path data/UD_German-GSD --max-sentences 20000 --spacy-model de_dep_news_trf

# TRMOR:
python analyze_UD.py --analyzer trmor --max-sentences 10000

# TRMORPH:
python analyze_UD.py --analyzer trmorph --max-sentences 10000 --trmorph-mode majority

# VNLP:
python analyze_UD.py --analyzer vnlp --max-sentences 10000
python run_and_merge_vnlp.py

# Zemberek:
# python analyze_UD.py --analyzer zemberek --max-sentences 10000
# Run parallel chunk analysis and merge results for zemberek
python run_and_merge_zemberek.py

# Zeyrek:
# pip install zeyrek
python analyze_UD.py --analyzer zeyrek --max-sentences 10000
