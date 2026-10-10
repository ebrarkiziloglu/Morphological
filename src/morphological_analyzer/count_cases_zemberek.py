"""
Turkish Case Counting Script (Zemberek + TULAP POS tagger)

This script:
1. Loads a monolingual Turkish pretraining dataset (plain text, one sentence per line,
   or an Arrow dataset saved with `save_to_disk`)
2. Analyzes every sentence with Zemberek (sentence-level tokenization, morphological
   analysis and contextual disambiguation), in parallel CPU worker processes
3. POS-tags the same tokens with the TULAP POS tagger (Buseak/pos_tagger_3112_v3), on GPU
   if available -- the same tagger used by
   pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py
4. Picks one analysis per token:
     - Zemberek's disambiguated (context-aware) best analysis, if it agrees with the POS tag
     - otherwise the highest-ranked Zemberek analysis that agrees with the POS tag
     - otherwise Zemberek's disambiguated best analysis
   Unknown tokens with an apostrophe (Google'ın, Trump'a) fall back to a suffix regex.
5. Counts the case suffixes of the chosen analyses over the whole dataset:
   Accusative (Acc), Dative (Dat), Genitive (Gen), Locative (Loc), Ablative (Abl),
   Nominative (Nom = nominal token without any case suffix), plus Instrumental (Ins),
   relative -ki (Ki), no_case and no_analysis
6. Reports and saves statistics
"""

import os
import re
import sys
import json
import time
import argparse
import logging
import typing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

# zemberek-python expects the removed typing.io module on some Python versions
typing.io = typing  # type: ignore[attr-assign]
sys.modules.setdefault("typing.io", typing)
logging.getLogger("zemberek").setLevel(logging.WARNING)

# Drop this script's directory from sys.path: the local zemberek.py would otherwise shadow
# the zemberek-python package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != _THIS_DIR]

REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
TULAP_DIR = os.path.join(REPO_ROOT, "Turkish_POS", "tools", "tulap")

CASES = ['Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Nom']
CASE_NAMES = {
    'Acc': 'accusative', 'Dat': 'dative', 'Gen': 'genitive',
    'Loc': 'locative', 'Abl': 'ablative', 'Nom': 'nominative',
}
# Zemberek case morphemes (Nom is not a morpheme: it is a nominal without any of these)
CASE_MORPHEMES = {'Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins'}
NOMINAL_POS = {'Noun', 'Pron'}
EXTRA_CATEGORIES = ['Ins', 'Ki', 'no_case', 'no_analysis']

POS_TAGS = {
    "Noun", "Adj", "Verb", "Pron", "Adv", "Num", "Postp",
    "Conj", "Det", "Ques", "Interj", "Punc",
}

# UD POS tag (TULAP) -> Zemberek primary/last-IG POS (as in remove_cases_from_tr_nllb.py)
UD_TO_ZEMBEREK = {
    "NOUN": {"Noun"},
    "PROPN": {"Noun"},
    "VERB": {"Verb"},
    "ADJ": {"Adj"},
    "ADV": {"Adv"},
    "PRON": {"Pron"},
    "ADP": {"Postp"},
    "CCONJ": {"Conj"},
    "SCONJ": {"Conj"},
    "DET": {"Det"},
    "NUM": {"Num"},
    "AUX": {"Verb"},
    "PART": {"Postp", "Conj"},
    "INTJ": {"Interj"},
    "PUNCT": {"Punc"},
}

# Case suffix after an apostrophe, for proper nouns / numbers Zemberek does not know
# (Google'ın, Trump'a, 1.000'den). Optional relative -ki after Loc/Gen (İstanbul'daki).
APOSTROPHE_SUFFIX_REGEX = [
    ('Abl', re.compile(r"^n?[dt][ae]n$")),
    ('Loc', re.compile(r"^n?[dt][ae](ki)?$")),
    ('Gen', re.compile(r"^[yn]?[ıiuü]n(ki)?$")),
    ('Dat', re.compile(r"^[yn]?[ae]$")),
    ('Acc', re.compile(r"^[yn]?[ıiuü]$")),
]
_LETTER_RE = re.compile(r"[^\W\d_]")


# ============================================================
# 1. Data Loading
# ============================================================

def load_sentences(tr_path, load_from_arrow=False, max_samples=None):
    """Load Turkish sentences from a text file or an Arrow dataset."""
    if load_from_arrow:
        from datasets import load_from_disk
        dataset = load_from_disk(tr_path)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        column = 'tr' if 'tr' in dataset.column_names else 'text'
        if column not in dataset.column_names:
            raise ValueError(f"Dataset has no 'tr' or 'text' column: {dataset.column_names}")
        return [s for s in dataset[column] if s and s.strip()]

    sentences = []
    with open(tr_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def turkish_lower(word):
    """Python's lower() maps 'I' -> 'i' and 'İ' -> 'i̇'; fix both for Turkish."""
    return word.replace('I', 'ı').replace('İ', 'i').lower()


# ============================================================
# 2. Zemberek Sentence Analysis (CPU worker processes)
# ============================================================

# Per-process TurkishMorphology instance (loaded once in each worker)
_MORPHOLOGY = None


def _init_morphology():
    global _MORPHOLOGY
    if _MORPHOLOGY is None:
        from zemberek import TurkishMorphology
        _MORPHOLOGY = TurkishMorphology.create_with_defaults()
    return _MORPHOLOGY


def _parse_info(parse):
    """Compact, picklable summary of a Zemberek parse."""
    tags = [md.morpheme.id_ for md in parse.morpheme_data_list]
    last_pos = next((t for t in reversed(tags) if t in POS_TAGS), None)
    return {
        'primary': parse.item.primary_pos.short_form,
        'last_pos': last_pos,
        'secondary': parse.item.secondary_pos.short_form,
        'cases': [t for t in tags if t in CASE_MORPHEMES],
        'n_rel': sum(1 for t in tags if t == 'Rel'),
    }


def _analyze_sentence(morphology, sentence):
    """
    Tokenize, analyze and disambiguate one sentence.

    Returns a list of tokens: (surface, parse_infos, best_index), where parse_infos are
    Zemberek's known analyses (in Zemberek's order) and best_index points at the
    disambiguated analysis (None if the token is unknown).
    """
    word_analyses = morphology.analyze_sentence(sentence)
    try:
        best = morphology.disambiguate(sentence, word_analyses).best_analysis()
        if len(best) != len(word_analyses):
            best = [None] * len(word_analyses)
    except Exception:
        best = [None] * len(word_analyses)

    tokens = []
    for word_analysis, best_parse in zip(word_analyses, best):
        parses = [p for p in word_analysis if not p.is_unknown()]
        best_index = None
        if parses:
            best_index = 0
            if best_parse is not None and not best_parse.is_unknown():
                best_str = best_parse.format_string()
                for i, p in enumerate(parses):
                    if p.format_string() == best_str:
                        best_index = i
                        break
        tokens.append((word_analysis.inp, [_parse_info(p) for p in parses], best_index))
    return tokens


def _analyze_chunk(sentences):
    """Worker: analyze a chunk of sentences."""
    morphology = _init_morphology()
    results = []
    for sentence in sentences:
        try:
            results.append(_analyze_sentence(morphology, sentence))
        except Exception:
            results.append(None)
    return results


# ============================================================
# 3. POS Tagging (TULAP, GPU)
# ============================================================

def load_pos_tagger(device):
    sys.path.insert(0, TULAP_DIR)
    from postagger import load_tulap_tagger
    return load_tulap_tagger(device)


def pos_tag_batch(tagger, batch_words):
    """
    POS-tag a batch of pre-tokenized sentences (lists of Zemberek tokens).

    Each word gets the tag predicted for its first sub-word piece
    (as in remove_cases_from_tr_nllb.pos_tag_words, but batched).
    """
    import torch

    non_empty = [i for i, words in enumerate(batch_words) if words]
    all_tags = [[] for _ in batch_words]
    if not non_empty:
        return all_tags

    inputs = [batch_words[i] for i in non_empty]
    encoded = tagger.tokenizer(
        inputs,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = tagger.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            **({"token_type_ids": encoded["token_type_ids"]} if "token_type_ids" in encoded else {}),
        ).logits
    pred_ids = torch.argmax(logits, dim=2).tolist()

    for b, i in enumerate(non_empty):
        words = batch_words[i]
        tags = [None] * len(words)
        for idx, word_id in enumerate(encoded.word_ids(b)):
            if word_id is None or tags[word_id] is not None:
                continue
            tags[word_id] = tagger.id2label.get(pred_ids[b][idx], "X")
        # Words cut off by truncation get "X" (no POS constraint)
        all_tags[i] = [t or "X" for t in tags]
    return all_tags


# ============================================================
# 4. Analysis Selection and Case Counting
# ============================================================

def parse_matches_pos(info, ud_pos):
    """Does a Zemberek parse agree with the UD POS tag? (as in remove_cases_from_tr_nllb.py)"""
    if not ud_pos or ud_pos == "X":
        return False
    targets = UD_TO_ZEMBEREK.get(ud_pos, set())
    if not targets:
        return False
    labels = {info['primary']}
    if info['last_pos']:
        labels.add(info['last_pos'])
    if ud_pos == "PROPN":
        return "Noun" in labels and info['secondary'] == "Prop"
    if ud_pos == "NOUN":
        return "Noun" in labels and info['secondary'] != "Prop"
    return bool(labels & targets)


def choose_parse(parse_infos, best_index, ud_pos):
    """
    Pick the analysis for a token.

    Returns (parse_info, how) where how is one of:
      'disambiguated' -- Zemberek's contextual best agrees with the POS tag
      'pos_override'  -- the POS tag picked another analysis than Zemberek's best
      'no_pos_match'  -- no analysis agrees with the POS tag; Zemberek's best is used
    """
    best = parse_infos[best_index]
    if parse_matches_pos(best, ud_pos):
        return best, 'disambiguated'
    for info in parse_infos:
        if parse_matches_pos(info, ud_pos):
            return info, 'pos_override'
    return best, 'no_pos_match'


def apostrophe_fallback_case(surface):
    """Case of an unknown token from the suffix after its apostrophe (Google'ın -> Gen)."""
    parts = re.split(r"['’‘]", surface, maxsplit=1)
    if len(parts) != 2 or not parts[0]:
        return None
    suffix = turkish_lower(parts[1])
    for case, pattern in APOSTROPHE_SUFFIX_REGEX:
        if pattern.match(suffix):
            return case
    return None


class CaseCounter:
    """Accumulates case counts over the dataset."""

    def __init__(self):
        self.suffix_counts = Counter()      # case suffix occurrences (a token can carry several)
        self.token_counts = Counter()       # tokens per category (tokens with >1 case counted in each)
        self.words_by_category = defaultdict(Counter)
        self.selection_counts = Counter()
        self.n_tokens = 0
        self.n_sentences = 0
        self.n_failed_sentences = 0

    def _add(self, category, word, n=1):
        self.suffix_counts[category] += n
        self.token_counts[category] += 1
        self.words_by_category[category][word] += 1

    def add_sentence(self, tokens, pos_tags):
        self.n_sentences += 1
        if tokens is None:
            self.n_failed_sentences += 1
            return
        for (surface, parse_infos, best_index), ud_pos in zip(tokens, pos_tags):
            # Skip punctuation and pure numbers (no letters), as in the earlier version
            if not _LETTER_RE.search(surface):
                continue
            self.n_tokens += 1
            word = turkish_lower(surface)

            if not parse_infos:
                case = apostrophe_fallback_case(surface)
                if case:
                    self.selection_counts['apostrophe_fallback'] += 1
                    self._add(case, word)
                else:
                    self.selection_counts['no_analysis'] += 1
                    self._add('no_analysis', word)
                continue

            info, how = choose_parse(parse_infos, best_index, ud_pos)
            self.selection_counts[how] += 1

            case_counts = Counter(info['cases'])
            for case, n in case_counts.items():
                self._add(case, word, n)
            if info['n_rel']:
                self._add('Ki', word, info['n_rel'])
            if not case_counts:
                if info['last_pos'] in NOMINAL_POS:
                    self._add('Nom', word)
                else:
                    self._add('no_case', word)


# ============================================================
# 5. Statistics and Reporting
# ============================================================

def build_summary_table(counter):
    """Build a markdown table with suffix and token counts per case."""
    categories = CASES + EXTRA_CATEGORIES
    case_suffixes = sum(counter.suffix_counts[c] for c in CASES)

    lines = [
        "| Case | Suffix count | Tokens | % of all tokens | % of the 6 cases | Unique words |",
        "|------|--------------|--------|-----------------|------------------|--------------|",
    ]
    for c in categories:
        suf, tok = counter.suffix_counts[c], counter.token_counts[c]
        pct_all = tok / counter.n_tokens * 100 if counter.n_tokens else 0
        pct_case = f"{suf / case_suffixes * 100:.2f}%" if c in CASES and case_suffixes else "-"
        label = f"{CASE_NAMES[c].capitalize()} (<{c}>)" if c in CASES else c
        uniq = len(counter.words_by_category[c])
        lines.append(f"| {label} | {suf:,} | {tok:,} | {pct_all:.2f}% | {pct_case} | {uniq:,} |")
    lines.append(f"| **Total tokens** | - | {counter.n_tokens:,} | 100.00% | - | - |")
    return "\n".join(lines)


def print_statistics(counter):
    print("=" * 60)
    print("CASE COUNT STATISTICS (Zemberek + TULAP POS)")
    print("=" * 60)
    print(f"\n📊 Basic Statistics:")
    print(f"   Sentences:                       {counter.n_sentences:,}")
    print(f"   Failed sentences:                {counter.n_failed_sentences:,}")
    print(f"   Word tokens:                     {counter.n_tokens:,}")
    print(f"\n🔎 Analysis selection:")
    for how in ['disambiguated', 'pos_override', 'no_pos_match', 'apostrophe_fallback', 'no_analysis']:
        n = counter.selection_counts[how]
        pct = n / counter.n_tokens * 100 if counter.n_tokens else 0
        print(f"   {how:<32} {n:>12,} ({pct:.2f}%)")
    print(f"\n📊 Case Counts:\n")
    print(build_summary_table(counter))
    print("\n   Suffix count: number of case suffixes (a token like 'evdekinden' has Loc + Abl).")
    print("   Tokens: number of tokens carrying the case. Nom = nominal token without a case suffix.")
    print("\n" + "=" * 60)


def save_results(output_dir, args, counter, top_k=50):
    """Save the counts as JSON and a markdown report with the most frequent words per case."""
    os.makedirs(output_dir, exist_ok=True)
    categories = CASES + EXTRA_CATEGORIES

    json_path = os.path.join(output_dir, "case_counts.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'tr_path': args.tr_path,
            'max_samples': args.max_samples,
            'method': 'zemberek_disambiguation+tulap_pos',
            'num_sentences': counter.n_sentences,
            'failed_sentences': counter.n_failed_sentences,
            'total_tokens': counter.n_tokens,
            'suffix_counts': {c: counter.suffix_counts[c] for c in categories},
            'token_counts': {c: counter.token_counts[c] for c in categories},
            'unique_words': {c: len(counter.words_by_category[c]) for c in categories},
            'selection_counts': dict(counter.selection_counts),
        }, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, "case_counts.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Case Counts (Zemberek + TULAP POS)\n\n")
        f.write(f"**Dataset:** `{args.tr_path}`\n\n")
        f.write(f"**Sentences:** {counter.n_sentences:,}\n\n")
        f.write(build_summary_table(counter) + "\n\n")
        f.write("**Analysis selection:** " + ", ".join(
            f"{k}: {v:,}" for k, v in counter.selection_counts.most_common()) + "\n\n---\n\n")
        for c in categories:
            words = counter.words_by_category[c].most_common(top_k)
            f.write(f"## {c} — top {len(words)} most frequent words\n\n")
            if not words:
                f.write("*No words in this category*\n\n")
                continue
            f.write("| Word | Frequency |\n|------|-----------|\n")
            for w, n in words:
                f.write(f"| {w} | {n:,} |\n")
            f.write("\n")

    print(f"\n✅ Results saved to:\n   - {json_path}\n   - {md_path}")


# ============================================================
# 6. Logging
# ============================================================

# Default log directory: src/logs (next to the other job logs)
LOG_DIR = os.path.join(os.path.dirname(_THIS_DIR), "logs")


class Tee:
    """Write everything to both the terminal and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


# ============================================================
# 7. Main Pipeline
# ============================================================

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main(args):
    print("🚀 Starting Zemberek + TULAP POS case counting pipeline...")

    print("\n📁 Step 1: Loading dataset...")
    sentences = load_sentences(args.tr_path, load_from_arrow=args.arrow, max_samples=args.max_samples)
    print(f"   Loaded {len(sentences):,} sentences")

    print(f"\n🏷️  Step 2: Loading TULAP POS tagger on {args.device}...")
    tagger = load_pos_tagger(args.device)

    print("\n🔬 Step 3: Zemberek sentence analysis (CPU) + POS tagging + case counting...")
    print(f"   Zemberek workers: {args.num_workers}, chunk size: {args.chunk_size}, "
          f"POS batch size: {args.batch_size}")
    chunks = [sentences[i:i + args.chunk_size] for i in range(0, len(sentences), args.chunk_size)]
    counter = CaseCounter()
    start = time.time()

    def process(chunk_tokens):
        for b in range(0, len(chunk_tokens), args.batch_size):
            batch = chunk_tokens[b:b + args.batch_size]
            batch_words = [[t[0] for t in toks] if toks else [] for toks in batch]
            batch_tags = pos_tag_batch(tagger, batch_words)
            for toks, tags in zip(batch, batch_tags):
                counter.add_sentence(toks, tags)

    def report_progress(i):
        if i % args.log_every == 0 or i == len(chunks):
            elapsed = time.time() - start
            done = counter.n_sentences
            rate = done / elapsed if elapsed else 0
            eta = (len(sentences) - done) / rate if rate else 0
            print(f"   {done:,}/{len(sentences):,} sentences "
                  f"({rate:.1f} sent/s, ETA {eta / 60:.1f} min)", flush=True)

    if args.num_workers <= 1:
        for i, chunk in enumerate(chunks, 1):
            process(_analyze_chunk(chunk))
            report_progress(i)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_morphology) as executor:
            for i, chunk_tokens in enumerate(executor.map(_analyze_chunk, chunks), 1):
                process(chunk_tokens)
                report_progress(i)

    print_statistics(counter)

    if not args.no_save:
        save_results(args.output_dir, args, counter)

    return counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count Turkish case suffixes in a monolingual pretraining dataset "
                    "using Zemberek + the TULAP POS tagger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tr-path", type=str, default="data/NLLB_10M_subset/NLLB_10M.tr",
                        help="Path to Turkish text file (one sentence per line) or Arrow dataset")
    parser.add_argument("--arrow", action="store_true", help="Load --tr-path as an Arrow dataset")
    parser.add_argument("--dataset_name", type=str, default="NLLB_10M_tr", help="Name of the dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of sentences (None for all)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                        help="Number of parallel Zemberek processes (<=1 for sequential)")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Sentences per Zemberek worker task")
    parser.add_argument("--batch-size", type=int, default=64, help="Sentences per POS tagger batch")
    parser.add_argument("--device", type=str, default="cuda" if _cuda_available() else "cpu",
                        help="Device for the POS tagger")
    parser.add_argument("--log-every", type=int, default=20, help="Print progress every N chunks")
    parser.add_argument("--output-dir", type=str, default="morphological_analyzer/nllb_TR_dataset_zemberek_counts",
                        help="Directory to save results")
    parser.add_argument("--no-save", action="store_true", help="Do not save results to files")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Log file (default: src/logs/count_cases_zemberek_<dataset_name>.log)")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset_name)

    # Mirror all terminal output into the log file
    if args.log_file is None:
        args.log_file = os.path.join(LOG_DIR, f"count_cases_zemberek_{args.dataset_name}.log")
    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
    log_handle = open(args.log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)
    print(f"📝 Logging to {os.path.abspath(args.log_file)}")

    start_time = time.time()
    main(args)
    samples_info = f"{args.max_samples} samples" if args.max_samples else "all samples"
    print(f"Time taken: {time.time() - start_time:.4f} seconds for {samples_info}")
