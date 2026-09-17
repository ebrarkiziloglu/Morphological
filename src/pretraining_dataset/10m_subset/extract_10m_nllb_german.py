#!/usr/bin/env python3
"""Extract a German NLLB subset matched to the 10M English NLLB byte budget.

Like extract_10m_nllb_trilingual.py this reads local NLLB text and writes a
10M-scale subset. Sampling follows extract_de_babylm.py: scale the English
length stats by a byte-premium, then draw sentences from length bins so the
German subset matches the target log-normal distribution and UTF-8 byte count.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm


# this file: Morphological/src/pretraining_dataset/10m_subset/...
REPO_ROOT = Path(__file__).resolve().parents[3]
PRETRAINING_DIR = Path(__file__).resolve().parents[1]
if str(PRETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_DIR))

NLLB_DE_DIR = REPO_ROOT / "data" / "NLLB_german"
TXT_SAVE_DIR = REPO_ROOT / "data" / "NLLB_10M_subset"
DEFAULT_OUTPUT_PATH = TXT_SAVE_DIR / "NLLB_10M.de"

# NLLB is already sentence-segmented. Use the same quality floor as the
# English 10M NLLB extract (MIN_WORDS=10); do not apply BabyLM's 20-word /
# 100-char cuts, which were for long paragraphs (mean ~219 chars).
MIN_WORDS = 10
MAX_CHARS = 1500
BIN_WIDTH = 10
POOL_MULTIPLIER = 5
SEED = 42


def get_log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    return (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(
        -((math.log(x) - mu) ** 2) / (2 * sigma**2)
    )


def iter_german_lines(input_dir: Path):
    """Yield stripped German sentences from NLLB text files in input_dir."""
    files = sorted(input_dir.glob("*.de"))
    if not files:
        files = sorted(p for p in input_dir.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"No German NLLB files found in {input_dir}")

    for path in files:
        print(f"Reading {path}...")
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                text = raw.replace("\n", " ").strip()
                if text:
                    yield text


def extract_german_subset(
    output_path: Path,
    original_bytes: int,
    original_mean: float,
    original_median: float,
    original_std: float,
    byte_premium: float,
    input_dir: Path = NLLB_DE_DIR,
    token: str | None = None,
) -> None:
    del token  # accepted for CLI compatibility with the sbatch wrapper

    if not input_dir.exists():
        raise FileNotFoundError(f"NLLB German directory not found: {input_dir}")

    target_bytes = int(original_bytes * byte_premium)
    target_mean = original_mean * byte_premium
    target_median = original_median * byte_premium
    target_std = original_std * byte_premium

    print(f"Byte premium: {byte_premium}")
    print(f"{'Metric':<20}{'Original':>15}{'Target':>15}")
    print("-" * 50)
    print(f"{'Bytes':<20}{original_bytes:>15,}{target_bytes:>15,}")
    print(f"{'Mean':<20}{original_mean:>15,.2f}{target_mean:>15,.2f}")
    print(f"{'Median':<20}{original_median:>15,.2f}{target_median:>15,.2f}")
    print(f"{'Std':<20}{original_std:>15,.2f}{target_std:>15,.2f}")

    if target_mean <= target_median:
        raise ValueError(
            f"Log-normal params require mean > median, got mean={target_mean}, "
            f"median={target_median}"
        )

    mu = math.log(target_median)
    sigma = math.sqrt(2 * math.log(target_mean / target_median))

    print("Collecting candidate sentences to match the target length distribution...")
    pool_bins: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    pool_total_bytes = 0
    n_seen = 0
    n_kept = 0

    for text in tqdm(iter_german_lines(input_dir), desc="Buffering candidates"):
        n_seen += 1
        char_len = len(text)
        word_count = len(text.split())
        if word_count < MIN_WORDS or char_len > MAX_CHARS:
            continue

        line = text + "\n"
        line_bytes = len(line.encode("utf-8"))
        bin_idx = char_len // BIN_WIDTH
        pool_bins[bin_idx].append((line, char_len, line_bytes))
        pool_total_bytes += line_bytes
        n_kept += 1

        if pool_total_bytes > target_bytes * POOL_MULTIPLIER:
            break

    print(
        f"\nScanned {n_seen:,} lines, kept {n_kept:,} candidates "
        f"({pool_total_bytes / 1024 / 1024:.2f} MB). Now sampling..."
    )
    if not pool_bins:
        raise RuntimeError("No candidate sentences remained after length filters.")

    bin_probs = {}
    for bin_idx in pool_bins:
        bin_center = (bin_idx * BIN_WIDTH) + (BIN_WIDTH / 2)
        bin_probs[bin_idx] = get_log_normal_pdf(bin_center, mu, sigma)

    random.seed(SEED)
    sampled_lines: list[str] = []
    sampled_chars: list[int] = []
    current_bytes = 0

    for bin_idx in pool_bins:
        random.shuffle(pool_bins[bin_idx])

    available_bins = list(pool_bins.keys())

    with tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Sampling matches") as pbar:
        while current_bytes < target_bytes and available_bins:
            current_total_prob = sum(bin_probs[b] for b in available_bins)
            if current_total_prob == 0:
                break

            r = random.random() * current_total_prob
            cum_prob = 0.0
            selected_bin = available_bins[-1]
            for b in available_bins:
                cum_prob += bin_probs[b]
                if r <= cum_prob:
                    selected_bin = b
                    break

            line, char_len, line_bytes = pool_bins[selected_bin].pop()
            sampled_lines.append(line)
            sampled_chars.append(char_len)

            current_bytes += line_bytes
            pbar.update(line_bytes)

            if not pool_bins[selected_bin]:
                available_bins.remove(selected_bin)

    random.shuffle(sampled_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving exactly {current_bytes:,} bytes to {output_path}...")
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

    mean_chars = statistics.mean(sampled_chars) if sampled_chars else 0
    median_chars = statistics.median(sampled_chars) if sampled_chars else 0
    std_chars = statistics.stdev(sampled_chars) if len(sampled_chars) > 1 else 0

    print("\n=== SAMPLING COMPLETE ===")
    print(f"Total samples written: {len(sampled_lines):,}")
    print(f"Actual bytes written:  {current_bytes:,}")
    print(f"Target bytes were:     {target_bytes:,}")
    print(f"Difference:            {current_bytes - target_bytes:,} bytes")

    print("\n--- Achieved Length Distribution (Characters) ---")
    print(f"Mean:   {mean_chars:.2f} (Target: ~{target_mean:.2f})")
    print(f"Median: {median_chars:.2f} (Target: ~{target_median:.2f})")
    print(f"StdDev: {std_chars:.2f} (Target: ~{target_std:.2f})")
    if sampled_chars:
        print(f"Min:    {min(sampled_chars)}")
        print(f"Max:    {max(sampled_chars)}")
    print(f"\nSaved locally to: {output_path}")

    if current_bytes < target_bytes:
        print(
            f"WARNING: only reached {current_bytes:,} bytes "
            f"(target {target_bytes:,}). Collect a larger pool or relax filters."
        )


def analyze_german_cases(dataset_path: Path, spacy_model: str = "de_core_news_lg") -> None:
    print(f"\n--- Starting morphological analysis using spaCy ({spacy_model}) ---")
    try:
        import spacy
    except ImportError:
        print("spaCy is not installed. Please install it using 'pip install spacy'.")
        return

    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        print(f"Model '{spacy_model}' not found. Downloading...")
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
        nlp = spacy.load(spacy_model)

    print(f"Reading dataset from {dataset_path}...")
    with dataset_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    case_counts: Counter[str] = Counter()
    total_tokens = 0
    total_words_with_case = 0

    print(f"Analyzing {len(lines)} sentences...")
    disable_pipes = ["parser", "ner", "attribute_ruler", "lemmatizer"]
    if "trf" in spacy_model:
        spacy.prefer_gpu()

    for doc in tqdm(
        nlp.pipe(lines, batch_size=1000, disable=disable_pipes),
        total=len(lines),
        desc="Analyzing Cases",
    ):
        for token in doc:
            total_tokens += 1
            cases = token.morph.get("Case")
            if cases:
                case_counts[cases[0]] += 1
                total_words_with_case += 1

    print("\n=== Case Occurrences ===")
    for case, count in case_counts.most_common():
        percentage = (count / total_words_with_case) * 100 if total_words_with_case else 0
        print(f"{case}: {count:,} ({percentage:.2f}% of words with case)")

    print(f"\nTotal tokens analyzed: {total_tokens:,}")
    print(f"Total tokens with case: {total_words_with_case:,}")

    results = {
        "total_tokens": total_tokens,
        "total_tokens_with_case": total_words_with_case,
        "case_counts": dict(case_counts),
        "case_percentages": {
            case: (count / total_words_with_case) * 100
            for case, count in case_counts.items()
        },
    }
    stats_path = Path(str(dataset_path) + ".case_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nCase statistics saved to {stats_path}")


def analyze_german_distribution(
    dataset_path: Path, dataset_name: str = "NLLB_10M_de"
) -> None:
    from dataset_analysis_helper import analyze_single_language_dataset

    print(f"\n--- Analyzing distributional properties of {dataset_path} ---")
    with dataset_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    hf_dataset = Dataset.from_dict({"text": lines})
    analyze_single_language_dataset(hf_dataset, dataset_name=dataset_name, language="de")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and analyze a German subset of NLLB matched to the 10M English NLLB stats"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (accepted for CLI compatibility; local NLLB files are used)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip extraction and only run spaCy / distribution analysis",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="de_core_news_lg",
        help="spaCy German model to use",
    )
    parser.add_argument(
        "--original-bytes",
        type=int,
        required=True,
        help="Original (English 10M NLLB) byte count to scale",
    )
    parser.add_argument(
        "--original-mean",
        type=float,
        required=True,
        help="Original mean sentence length in characters",
    )
    parser.add_argument(
        "--original-median",
        type=float,
        required=True,
        help="Original median sentence length in characters",
    )
    parser.add_argument(
        "--original-std",
        type=float,
        required=True,
        help="Original std of sentence length in characters",
    )
    parser.add_argument(
        "--byte-premium",
        type=float,
        required=True,
        help="German/English byte-premium used to scale targets",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=NLLB_DE_DIR,
        help="Directory containing NLLB German text files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output text file for the German subset",
    )
    args = parser.parse_args()

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        extract_german_subset(
            output_path,
            args.original_bytes,
            args.original_mean,
            args.original_median,
            args.original_std,
            args.byte_premium,
            input_dir=args.input_dir,
            token=args.token,
        )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Output file {output_path} does not exist; run extraction first."
        )

    analyze_german_cases(dataset_path=output_path, spacy_model=args.spacy_model)
    analyze_german_distribution(dataset_path=output_path, dataset_name="NLLB_10M_de")


if __name__ == "__main__":
    main()
