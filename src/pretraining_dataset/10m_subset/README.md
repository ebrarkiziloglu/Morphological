# Pre-training 10M subsets

Scripts that carve fixed-size pre-training corpora from parallel NLLB (and, previously, HPLT) text.

The **current** setup is NLLB: a trilingual English–Turkish–Chinese subset of about **10 million English words**, plus a German subset matched to that English set by **UTF-8 byte budget** (byte-premium scaled). Downstream phase-1 training reads the one-sentence-per-line files under `data/NLLB_10M_subset/`.

Run Python from the **repo root** (`Morphological/`). The German script adds `src/pretraining_dataset` to `sys.path` itself; the trilingual and HPLT scripts import `dataset_analysis_helper`, so set `PYTHONPATH` as shown below. Each `.py` file has a matching SLURM wrapper (`.sh`) in this directory.

## Current scripts

### `extract_10m_nllb_trilingual.py` — EN / TR / ZH

Joins NLLB **EN–TR** and **TR–ZH** on shared Turkish sentences, then samples until ~10M English words.

1. Load TR–ZH into a Turkish → Chinese map (first occurrence of each Turkish line wins).
2. Scan EN–TR and keep one triple per shared Turkish sentence.
3. Drop short / very long lines (see [Length filters](#length-filters)).
4. Shuffle with seed `42`.
5. Take examples in shuffled order until the cumulative English word count reaches 10M (`str.split()`).
6. Write parallel text plus an Arrow cache, then run `analyze_dataset` on EN–TR.

**Inputs** (aligned Moses-style files):

| Pair | Files |
| --- | --- |
| EN–TR | `data/NLLB_EN_TR/NLLB.en-tr.en`, `data/NLLB_EN_TR/NLLB.en-tr.tr` |
| TR–ZH | `data/NLLB_TR_CH/NLLB.tr-zh.tr`, `data/NLLB_TR_CH/NLLB.tr-zh.zh` |

**Outputs:**

| Path | Contents |
| --- | --- |
| `data/NLLB_10M_subset/NLLB_10M.{en,tr,zh}` | Parallel lines (same index = same triple) |
| `data/arrow/NLLB_10M_subset/` | Hugging Face Arrow cache of the subset |

Chinese is not word-counted (no reliable space tokenization); the 10M target is English-only.

```bash
PYTHONPATH=src/pretraining_dataset python src/pretraining_dataset/10m_subset/extract_10m_nllb_trilingual.py
```

### `extract_10m_nllb_german.py` — German (byte-premium matched)

German is **not** word-matched to 10M. English and German differ in bytes per “same” text, so the script scales the English 10M NLLB length statistics by a **byte-premium** and samples German sentences so the subset matches that target **log-normal length distribution** and **UTF-8 byte count**. The sampling procedure follows `babylm_subset/extract_de_babylm.py` (length bins of 10 characters, pool ~5× the target bytes).

After extraction it runs spaCy case analysis (`de_core_news_lg` by default) and `analyze_single_language_dataset`.

**Input:** German NLLB text in `data/NLLB_german/` (`*.de`, or any files in that directory).

**Output:** `data/NLLB_10M_subset/NLLB_10M.de` and `NLLB_10M.de.case_stats.json`.

The English 10M stats and German/English premium used in the SLURM wrapper:

```bash
python src/pretraining_dataset/10m_subset/extract_10m_nllb_german.py \
    --original-bytes 59652031 \
    --original-mean 125.25 \
    --original-median 114.00 \
    --original-std 52.08 \
    --byte-premium 1.0537171
```

`--analyze-only` skips sampling and only analyzes an existing `--output-path`. `--token` is accepted for CLI compatibility; extraction reads local files.

## Length filters

| | Trilingual NLLB | German NLLB | Legacy HPLT |
| --- | --- | --- | --- |
| Min words (`str.split()`) | EN and TR ≥ 10 | DE ≥ 10 | EN and TR ≥ 16 |
| Max characters | EN, TR, ZH ≤ 1500 | DE ≤ 1500 | EN and TR ≤ 1500 |

NLLB is already sentence-segmented. The German extractor does **not** use BabyLM’s 20-word / 100-char cuts (those were for long paragraphs).

## Legacy script

`extract_10m_hplt.py` is the older bilingual extractor: HPLT EN–TR, filter, shuffle (`seed=42`), sample to 10M English words. Prefer the NLLB scripts above.

**Inputs:** `data/arrow/HPLT_en_tr` if present, otherwise `data/HPLT_en_tr/HPLT.en-tr.{en,tr}`.

**Outputs:** `data/HPLT_10M_subset/HPLT_10M.{en,tr}` and `data/arrow/HPLT_10M_subset/`.

```bash
PYTHONPATH=src/pretraining_dataset python src/pretraining_dataset/10m_subset/extract_10m_hplt.py
```

## Byte-premium (German)

The German/English factor `1.0537171` is from Arnett, Chang, and Bergen (2024), [A Bit of a Problem: Measurement Disparities in Dataset Sizes Across Languages](https://arxiv.org/abs/2403.00686). Scaling the English byte count and character-length mean/median/std by this premium keeps the German subset comparable in information content rather than in raw word or line count.
