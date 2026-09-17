# POS tagging for Turkish

This directory compares off-the-shelf Turkish POS taggers on a small held-out set of 20 sentences. Tagger wrappers live under `tools/`. `main.ipynb` loads each tagger, runs it on the test set, and scores it with span-aligned metrics in `eval_align.py`.

## Test set

Sentences come from [`hasankursun/turkish-corpus-100b`](https://huggingface.co/datasets/hasankursun/turkish-corpus-100b) (Hugging Face, `pretrain` config). The first 50 samples were inspected by hand; 20 sentences were kept and written to `pos_testset.txt`.

Gold labels in `pos_testset_groundtruth.csv` were produced by running TULAP and then correcting tags (and some token boundaries) manually. The gold tokenization therefore follows TULAP: stems and clitics are often split, and some surface forms are truncated to stems.

| File | Description |
| --- | --- |
| `pos_testset.txt` | One sentence per line (20 sentences) |
| `pos_testset.csv` | Raw TULAP output (tokens + tags), before manual correction |
| `pos_testset_groundtruth.csv` | Manually corrected gold (`text;tokens;tags`, 312 gold tokens) |

**Caveat.** Because gold is TULAP-derived, TULAP has a built-in advantage: its tokenization matches gold, and remaining errors are tag-choice errors only. Treat the ranking as a practical comparison on this set, not as an unbiased UD benchmark.

## How evaluation works

Taggers do not share a tokenizer. TULAP splits clitics and often emits stems; VNLP tags surface words. Direct token-index comparison would be invalid.

`eval_align.py` maps each token to a character span in the original sentence, then projects tags by maximum span overlap:

- **Word-level accuracy** — gold tags projected onto predicted tokens (precision-oriented).
- **Gold-token recall** — predicted tags projected onto gold tokens.
- **Span-aligned F1** — harmonic mean of the two.

When tokenizations match (as with TULAP), the three scores coincide.

Reproduce from this directory:

```bash
# sanity-check that gold tokens can be located in the source text
python eval_align.py

# full comparison: load taggers, score against gold
jupyter notebook main.ipynb
```

Python extras used by the notebook are listed in `requirements.txt` (`vngrs-nlp` for VNLP). TULAP also needs `torch` and `transformers`.

## Results

Scores on the 20-sentence set (312 gold tokens). Timing is wall-clock for tagging only, on CPU, as reported in `main.ipynb`.

| Tagger | Accuracy | Recall | F1 | Time | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| [TULAP](https://tulap.cmpe.bogazici.edu.tr/demo/postagger) | 0.8910 (278/312) | 0.8910 (278/312) | 0.8910 | 0.95 s | evaluated |
| [VNLP](https://vnlp.readthedocs.io/en/latest/main_classes/part_of_speech_tagger.html) | 0.8400 (252/300) | 0.8077 (252/312) | 0.8235 | 72.45 s | evaluated |
| [turkish-pos-tagger](https://github.com/onuryilmaz/turkish-pos-tagger) | — | — | — | — | not scored |

### 1. TULAP

Boğaziçi TULAP demo and the Hugging Face model [`Buseak/pos_tagger_3112_v3`](https://huggingface.co/Buseak/pos_tagger_3112_v3). Local wrapper: `tools/tulap/postagger.py`.

Uses Universal POS tags (`NOUN`, `VERB`, `PROPN`, …). Tokenization often peels clitics and reports stems rather than full surface forms.

### 2. turkish-pos-tagger (Onur Yılmaz)

Cloned into `tools/turkish-pos-tagger`. NLTK Brill tagger trained on a Turkish UD treebank subset; ships a YAML-serialized model (`my_tagger.yaml`).

It was not scored here. The code is Python 2-era (cp1254 encoding, NLTK Brill YAML), the tagset is not UD (`Noun_Abl`, `Postp`, `Punc`, …), and tagging is applied to each whitespace token in isolation.

### 3. VNLP

[`vnlp.PoSTagger`](https://vnlp.readthedocs.io/en/latest/main_classes/part_of_speech_tagger.html) (`vngrs-nlp`). Also uses Universal POS tags, but on surface-word tokens, so predicted sequences are shorter than gold (300 vs 312 tokens). Span alignment is required.

## Conclusion

On this 20-sentence set, TULAP is both more accurate (89% F1 vs 82%) and much faster (~1 s vs ~72 s). That ranking is consistent with the gold being corrected TULAP output and with TULAP sharing gold tokenization. VNLP is still usable if a surface-word tokenizer is preferred. The Brill-based `turkish-pos-tagger` was not a viable candidate for this comparison.

The test set is small (20 sentences / 312 tokens). Differences should be read as a smoke test for tool choice, not as a published benchmark.
