# Morphological Cross-Language Transfer

Research project studying whether pretraining a causal LM on a morphologically rich language (Turkish) transfers case-marking knowledge to German better than pretraining on English or Chinese first.

## Idea

1. Pretrain GPT-2-style models on phase 1 (TR / EN / ZH / control variants) then phase 2 (German).
2. Evaluate case-marking knowledge via minimal-pair perplexity: for sentences with a Dative-case token, the model should prefer the gold (Dative) form over an Accusative-corrupted variant.
3. Compare transfer across source languages and ablations (shuffled word order, case-suffix-stripped Turkish, character-shifted text, reset embeddings, BabyLM/NLLB data sources).



## Layout

- `src/pretraining_dataset/` — build pretraining corpora (NLLB/HPLT/BabyLM subsets), plus ablation transforms (word shuffling, char shifting, Turkish case-suffix stripping).
- `src/tokenizer/` — train BPE/char tokenizers (`tokenizer_32k`, `tokenizer_40k`, `tokenizer_char`, ...).
- `src/pretraining/` — `train.py` (GPT-2 causal LM training) plus per-language/phase SLURM scripts.
- `src/morphological_analyzer/` — wrappers around Turkish/German morphological analyzers (Zemberek, Zeyrek, VNLP, TRmorph, SMOR, spaCy-DE) used to measure case distribution in corpora.
- `src/eval_dataset/` — Accusative/Dative minimal-pair datasets (regular, shifted, unconverted variants).
- `src/eval/` — `evaluate_ppl.py` / `evaluate_ppl_hf.py` (minimal-pair perplexity eval, single checkpoint or multi-seed/epoch sweep) plus `significance_tests.py`.
- `src/ud_dataset*.ipynb`, `src/ud_german_test_methodology.md` — building Dative-vs-Accusative minimal pairs from UD German-GSD.
- `data/` — corpora and derived subsets (gitignored; see subset READMEs, e.g. `src/pretraining_dataset/10m_subset/README.md`).
- `models/` — trained checkpoints / tokenizer artifacts and HF upload scripts (gitignored).
- `eval_results/` — aggregated perplexity results per epoch/model (`aggregated_eval_results.md`, `eval.ipynb`) and per-run CSVs/plots.
- `Turkish_POS/` — side comparison of off-the-shelf Turkish POS taggers (TULAP vs VNLP vs turkish-pos-tagger).
- `morphological_analyzer.sh`, `how-to-build-for-TRMOR.sh` — SLURM job / build script for running the Turkish/German morphological analyzers (TRMOR needs OpenFST built from source).



## Setup

```bash
conda activate morph2
# or
pip install -r requirements.txt
```

Requires conda environments with `sfst`/OpenFST on the cluster for the TRMOR/SMOR analyzers (see `how-to-build-for-TRMOR.sh`).

