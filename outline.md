# Project Outline: Cross-Lingual Transfer of Morphological Case Knowledge

~8 months of work, ~16 h/week. Items marked **[TODO]** will be filled in.

---

## 1. Objective and hypothesis

**Objective.** Detect whether morphological knowledge transfers across languages during sequential pretraining of small language models.

**Hypothesis.**

- Turkish is agglutinative and morphologically rich. Its noun cases are Nominative, Accusative, Dative, Locative, Ablative and Genitive.
- English has (almost) no nominal case marking.
- German has four cases (Nom, Acc, Dat, Gen), but they are realized very differently from Turkish: mostly on articles, pronouns and adjectives, with fusional rather than agglutinative marking.
- Despite these structural differences, we hypothesize that case knowledge learned from Turkish transfers to German. Pretraining on Turkish first should therefore help German case sensitivity more than pretraining on English first.
- Chinese is a second morphologically-poor control (no inflection, no whitespace segmentation), also very far away from the German and English semnatic similarities.

**Operationalization.** German case sensitivity is measured on Accusative↔Dative minimal pairs. The model should assign lower perplexity to the grammatical sentence than to the case-corrupted one.

---



## 2. Experimental design



### 2.1 Overview

Two-phase pretraining of decoder-only models:


| Phase   | Data                                 | Purpose                                       |
| ------- | ------------------------------------ | --------------------------------------------- |
| Phase 1 | One L1 ∈ {Turkish, English, Chinese} | Acquire L1-specific (morphological) knowledge |
| Phase 2 | Always German (L2)                   | Transfer test                                 |


Baseline: a German-only model, trained on the Phase 2 data with no Phase 1 (`char-de`).

### 2.2 Model and tokenizer

- **Architecture:** GPT-2-style decoder-only LM (`GPT2LMHeadModel`), trained from scratch. Code: `src/pretraining/train.py`.
  - 6 layers, 12 heads, hidden size 768
  - max sequence length 1024 (positions 1024)
  - weight-tied embedding and output head
- **Tokenizer:** a shared **character-level** tokenizer (`src/tokenizer/tokenizer_char`). It is a `WordLevel` model with a regex split into single characters, built by `train_char_tokenizer.py`.
  - Vocabulary: 2,442 symbols (all unique characters in a multilingual corpus) plus special tokens `[BOS] [PAD] [EOS] [UNK] ...`
  - One tokenizer for all languages, so the embedding table is shared across phases.
  - Character-level input prevents subword segmentation from hiding or leaking morphology, and removes vocabulary mismatch between Phase 1 and Phase 2.
  - Earlier BPE tokenizers (32k, 40k) exist in `src/tokenizer/` but the reported runs (`models/v3`, `char-*`) use the character tokenizer.
- **Optimization (both phases):**
  - AdamW, β = (0.9, 0.98), weight decay 0.1
  - cosine schedule with warmup of 1% of steps
  - grad-clip 1.0
  - bf16 autocast, FlashAttention-3 on H100 GPUs
  - `grad_acc = 2`
- **Phase 1:** 5 epochs, per-device batch size 16 (effective 32), LR 1e-4. Only the epoch-5 checkpoint is kept.
- **Phase 2:** 5 epochs, batch size 32 (effective 64), LR 5e-5. Checkpoints after every epoch; evaluation at epochs 1, 3, 5.
- **Seeds:** 5 (0–4) per condition. Phase 2 resumes from the Phase 1 checkpoint of the same seed.



### 2.3 Phase 1 data (L1)

- **Source:** NLLB, via a parallel EN–TR–ZH subset (`src/pretraining_dataset/10m_subset/extract_10m_nllb_trilingual.py`).
  - EN–TR is joined with TR–ZH on shared Turkish sentences, so every line is a true translation triple.
  - Filters: ≥10 words (EN, TR), ≤1500 characters.
  - Shuffled with seed 42, then sampled until the cumulative English word count reaches **10M words**.
  - Result: **469,497 parallel lines**, ≈59.7 MB English, 65.4 MB Turkish, 52.8 MB Chinese.
  - Chinese is not word-counted. The 10M target is defined on English only.
- Because the data is parallel, differences between L1 conditions are due to the language, not the content.
- Earlier versions of the corpus (HPLT, OpenSubtitles) were abandoned in favour of NLLB (sentence-segmented, parallel).



### 2.4 Phase 1 data variants (ablations)

**(a) Word-order shuffled.** Script: `src/pretraining_dataset/shuffle_words.py`.

- For each line, 30% of the tokens are selected. Each selected token is displaced by a distance drawn from a geometric distribution (p = 0.5, max 20), in a random direction.
- Tokens are re-sorted by their displaced position.
- Whitespace tokens are used for EN/TR; characters are used for ZH.
- Geometric displacements mimic the distribution of syntactic dependency lengths, so local syntax is destroyed while the vocabulary and the morphology inside words are kept.
- Seed 42.
- Purpose: isolate the contribution of word-internal morphology from word order and syntax.

**(b) Turkish morphology-stripped variants.** Script: `src/pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py`.

- Pipeline:
  1. POS tagging with the TULAP tagger (`Turkish_POS/`).
  2. Morphological analysis with Zemberek, disambiguated against the POS tag.
  3. Removal of selected morphemes.
  4. Word regeneration with Zemberek's generator.
- Variants (directory and model names in brackets):


| Variant                                               | What is removed                                                                                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Case-stripped (`stripped_case`)                       | All case markers: Acc, Dat, Loc, Abl, Gen, plus instrumental `-(y)lA`, equative `-cA`, and relative `-ki`. Plural and possessive are kept. |
| Postp-stripped (`stripped_postp`)                     | Everything in case-stripped, plus the postposition words themselves (için, ile, gibi, göre, kadar, sonra, ...). Lines get shorter.         |
| Nominal-inflection-stripped (`stripped_nominal_infl`) | All nominal inflection (case, plural, possessive, `-ki`, nominal predicate suffixes). All derivation is kept.                              |
| Nominal-stripped (`stripped_nominal`)                 | All nominal morphology, inflection and derivation. Nominalised verbs are cut back to the verb stem. Verbal inflection is kept.             |


- Examples of the stripping (`evdekinden → ev`, `kitabımı → kitabım / kitap`) are in the script docstring.
- Per-variant statistics (counts of removed morphemes, deleted postpositions) are written to `*.stats.txt`. The `evaluate_stripping.py` script checks the stripped files against the originals.
- The stripped variants also exist in word-shuffled form (`NLLB_10M_stripped*_shuffled_geometric_r0.3.tr`).
- Purpose: a causal intervention. If case suffixes in Turkish are what transfers, removing them should reduce the Turkish advantage.

**(c) Corpus case statistics.** A Turkish case-counting script (Zemberek + TULAP) measures how often each case occurs, so the amount of case signal in each corpus is quantified. German counts come from spaCy (`NLLB_10M.de.case_stats.json`).
Morphological analyzers explored (kept in `src/morphological_analyzer/`): Zemberek, Zeyrek, VNLP, TRmorph/TRMOR, SMOR, spaCy-DE. Zemberek was chosen for stripping.

### 2.5 Phase 2 data (German)

- **Source:** NLLB German side (`extract_10m_nllb_german.py`).
- **Size matching:** the German set is matched to the English 10M set by **UTF-8 byte budget**, not word count.
  - Byte premium: **German/English = 1.0537171**. The English subset is 59,652,031 bytes, so the German target is ≈ 62.86 MB (actual file: 62,856,465 bytes, 466,633 lines).
  - Sentence-length distribution (log-normal; English mean 125.25, median 114, std 52.08 characters) is scaled by the same premium.
  - Sampling uses 10-character length bins from a pool ~5× the target.
  - Rationale: the same content takes different numbers of bytes in different languages, and bytes are a fairer unit of "amount of information" than words. The byte premium follows Arnett, Chang & Bergen (2024), "A Bit of a Problem: Measurement Disparities in Dataset Sizes Across Languages" (arXiv:2403.00686). The value 1.0537171 is hard-coded in `extract_10m_nllb_german.sh`.
- A BabyLM-based German subset (`babylm_subset/`) was an earlier alternative to NLLB. Its original English reference size was 61,702,345 bytes.



### 2.6 Training configurations


| Config                     | Phase 1 data     | Between phases                                                      | Phase 2 |
| -------------------------- | ---------------- | ------------------------------------------------------------------- | ------- |
| config-1 (regular)         | regular L1       | –                                                                   | German  |
| config-2 (reset-embedding) | regular L1       | token embeddings (`wte`, tied `lm_head`) re-initialised: N(0, 0.02) | German  |
| config-3 (shuffled)        | word-shuffled L1 | –                                                                   | German  |


- **Why reset embeddings (config-2):** it destroys the lexical knowledge of L1 while keeping the transformer layers and positional embeddings. Any remaining transfer must then live in the non-lexical (structural) weights, so lexical overlap between L1 and German cannot explain it.
- Each configuration is run for L1 ∈ {TR, EN, ZH}. For Turkish, each is also run for each stripped variant (regular and shuffled).

---



## 3. Evaluation



### 3.1 Test set: German Acc↔Dat minimal pairs

Code: `src/eval_dataset/` (see its README). Built from **UD German-GSD** (train + dev + test, used only as a source of sentences; no training on it).

1. **Case dictionary.** `german_ud_lookup_dict_extract.py` scans Acc/Dat tokens in GSD, HDT, PUD and LIT. Each row is keyed by lemma, case, number, gender, degree, UPOS and (for adjectives) strong/weak inflection, and lists the observed surface forms with counts. No hand-written suffix rules are used.
2. **NP spans.** For every Acc or Dat token, the NP is found by walking the UD dependencies (`np_group_span.py`). **Every** token of the source case in that NP is rewritten using the dictionary, so the corrupted NP stays internally consistent (agreement between determiner, adjective and noun is preserved).
3. **Contractions** (`im`, `zum`, ...) are expanded and re-spliced.
4. **Filtering.** A pair is kept only if every token converts, the result differs from the original, and there are no duplicates. Spans that fail are logged as "unconverted" with a reason code.
5. **Two directions:**
  - Acc→Dat corruption: `accusative_dative_pairs.json`, **9,189** pairs. The gold sentence has Acc and the corrupt one has Dat.
  - Dat→Acc corruption: `dative_accusative_pairs.json`, **9,389** pairs. The gold sentence has Dat and the corrupt one has Acc.
6. **Dative categories** (Dat→Acc set only). Each pair has a `dative_type` label from `german_ud_dative_types.py`: preposition classes (`dative_prep_in`: 3,092; `_zu`: 1,089; `_mit`: 875; `_von`: 737; ...), core dative arguments (571), comparatives, etc. The prepositional datives are further grouped by which Turkish case they correspond to (Locative, Ablative, Instrumental, Mixed), aligned through the **UD Turkish/German PUD parallel treebank** (`pud_turkish_dative_align.py`, `pud_dative_parallel.ipynb`). Core dative arguments were translated to EN/TR (`translate_core_dative_tr.py`) and cleaned manually.
  - The Turkish-case grouping lets you ask: *does Turkish pretraining help most where German Dat corresponds to a Turkish Loc/Abl/Dat suffix?*
7. Earlier prototype (`ud_german_test_methodology.md`): only Dat→Acc, with a hand-written mapping for articles and pronouns. It was replaced by the dictionary approach.



### 3.2 Metric

- Each sentence is scored by its mean token NLL, and perplexity is exp(NLL), under the causal LM (`src/eval/evaluate_ppl.py`).
- A pair is **correct** if PPL(gold) ≤ PPL(corrupted). Ties favour gold, but they are practically absent for continuous PPL.
- **Accuracy** = fraction of correct pairs. Chance is 50%.
- Reported per seed, then as mean ± std over 5 seeds, overall and per category. Bootstrap CIs are available (`bootstrap_ci_mean`).
- Mean gold perplexity is also reported, as a general German LM-quality check.
- **Significance** (`src/eval/significance_tests.py`):
  - Between models: Welch t-test with seeds as the unit (n = 5).
  - Between categories: two-proportion z-test with examples as the unit.
  - TOST equivalence tests (δ = 1 and 2 pp), so a claim of "no difference" can be supported instead of only failing to reject.
- Eval runs: `src/eval/evaluate_*.sh` (one per config and L1). Results go to `eval_results/` (per-run CSVs, accuracy plots, `Models_Evaluations.xlsx`, `eval.ipynb`, HTML analyses `tr_en_transfer_analysis.html`, `tr_stripped_variants_analysis.html`).



### 3.3 Conditions evaluated

- Phase 1 only (before German), regular and shuffled, for TR, EN and ZH. This is the baseline for what a model knows without German training.
- Phase 2 (epochs 1, 3, 5): German-only baseline, and L1 ∈ {TR, EN, ZH} × {regular, reset-embedding, shuffled}.
- Turkish ablations: {stripped, stripped_postp, stripped_nominal_infl, stripped_nominal} × {regular, shuffled, reset-embedding}.
- Checkpoints are in `models/v3/` and have been uploaded to Hugging Face (`models/v3/upload_to_hf.py`).

---



## 4. Preliminary results (from `eval_results/`, epoch 5, mean over 5 seeds)

Accuracy of preferring the gold sentence:


| Model                     | Acc→Dat set | Dat→Acc set |
| ------------------------- | ----------- | ----------- |
| German only (no Phase 1)  | 80.9        | 83.8        |
| EN → DE                   | 85.2        | 85.7        |
| TR → DE                   | 84.9        | 85.5        |
| EN → DE, shuffled Phase 1 | 84.3        | 85.5        |
| TR → DE, shuffled Phase 1 | 84.6        | 85.5        |
| EN → DE, reset embeddings | 83.8        | 84.7        |
| TR → DE, reset embeddings | 83.5        | 85.1        |
| TR(stripped_case) → DE    | 84.5        | 85.5        |


Standard deviation across seeds is ≈ 0.2–0.5 pp.

**Observations so far** (to be confirmed with the significance tests):

- Any Phase 1 pretraining helps relative to German-only (≈ +4 pp on Acc→Dat). The gain shrinks from epoch 1 to epoch 5.
- The **hypothesised Turkish > English advantage is not visible in the aggregate**. TR and EN are within about 0.3 pp of each other, which is of the same order as the seed noise.
- Shuffling Phase 1 does not hurt much, and resetting embeddings costs ≈ 1–1.5 pp for both TR and EN. So most of the benefit looks language-agnostic: it comes from general LM pretraining, not from L1 morphology.
- `stripped_case` Turkish performs like regular Turkish.
- **[TODO]** Check per-category results (Locative/Ablative/Instrumental, core datives), epoch 1 (where the L1 differences are largest) and the ZH results. A possible narrative is a null or negative result, supported by TOST equivalence tests, plus an analysis of why the transfer signal is weak (small models, character-level input, 10M-word scale, German case marked mainly on closed-class words).

---



## 5. Open questions and gaps

1. **[TODO]** Exact count of training tokens or characters per phase. Lines and bytes are given above. The character-level token count follows from bytes, but is not stated anywhere.
2. **[TODO]** Why a 6-layer model with a 1024-character context? Justify the model size (parameter count ≈ 45M plus embeddings; compute it).
3. The test set is built from UD-GSD, which is also standard German text. It is not part of the training data, but check for sentence overlap with NLLB.
4. Limitations to mention: single architecture and scale, character-level tokenization, perplexity-based minimal pairs measure sensitivity but not productive use of case, German Dat/Acc only (not Gen/Nom).



## 6. Repository map


| Path                                          | Content                                                   |
| --------------------------------------------- | --------------------------------------------------------- |
| `src/pretraining_dataset/`                    | Corpus building, shuffling, Turkish stripping             |
| `src/tokenizer/`                              | Character tokenizer (and earlier BPE tokenizers)          |
| `src/pretraining/`                            | `train.py` and the SLURM scripts per L1, phase and config |
| `src/eval_dataset/`                           | Minimal-pair test set generation                          |
| `src/eval/`                                   | Perplexity evaluation, significance tests                 |
| `src/morphological_analyzer/`, `Turkish_POS/` | Analyzer and POS-tagger comparisons, case counting        |
| `eval_results/`                               | All results                                               |
| `models/v3/`                                  | Checkpoints                                               |


