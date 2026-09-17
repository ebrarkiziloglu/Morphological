#!/usr/bin/env python3
"""Extract a 10M-English-word trilingual EN-TR-ZH subset from NLLB.

Joins NLLB EN-TR and TR-ZH on shared Turkish sentences, filters by length,
shuffles, then samples until ~10M English words.
"""

from __future__ import annotations

import time
from pathlib import Path

from datasets import Dataset
from dataset_analysis_helper import analyze_dataset, word_count


REPO_ROOT = Path(__file__).resolve().parents[2]

EN_TR_EN = REPO_ROOT / "data" / "NLLB_EN_TR" / "NLLB.en-tr.en"
EN_TR_TR = REPO_ROOT / "data" / "NLLB_EN_TR" / "NLLB.en-tr.tr"
TR_ZH_TR = REPO_ROOT / "data" / "NLLB_TR_CH" / "NLLB.tr-zh.tr"
TR_ZH_ZH = REPO_ROOT / "data" / "NLLB_TR_CH" / "NLLB.tr-zh.zh"

TXT_SAVE_DIR = REPO_ROOT / "data" / "NLLB_10M_subset"
ARROW_SAVE_PATH = REPO_ROOT / "data" / "arrow" / "NLLB_10M_subset"

# Prior run with MIN_WORDS=16 kept only ~342k / 3.55M shared-TR hits (~9.36M EN
# words). Lowering the floor recovers shorter-but-usable sentences to reach 10M.
MIN_WORDS = 10
MAX_CHARS = 1500
TARGET_EN_WORDS = 10_000_000
SEED = 42
PROGRESS_EVERY = 1_000_000


def normalize_tr(text: str) -> str:
    """Normalize Turkish for exact-set membership (strip only)."""
    return text.strip()


def load_tr_to_zh(tr_path: Path, zh_path: Path) -> dict[str, str]:
    """Load TR-ZH into a Turkish -> Chinese map (first occurrence wins)."""
    tr_to_zh: dict[str, str] = {}
    n_seen = 0
    n_dup = 0
    t0 = time.time()

    print(f"Loading TR-ZH from:\n  {tr_path}\n  {zh_path}")
    with (
        tr_path.open("r", encoding="utf-8") as tr_f,
        zh_path.open("r", encoding="utf-8") as zh_f,
    ):
        for tr_line, zh_line in zip(tr_f, zh_f):
            n_seen += 1
            tr = normalize_tr(tr_line)
            if not tr:
                continue
            if tr in tr_to_zh:
                n_dup += 1
                continue
            tr_to_zh[tr] = zh_line.rstrip("\n")

            if PROGRESS_EVERY > 0 and n_seen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = n_seen / elapsed if elapsed > 0 else 0.0
                print(
                    f"  TR-ZH scanned {n_seen:,} | unique TR {len(tr_to_zh):,} | "
                    f"{rate:,.0f} lines/s",
                    flush=True,
                )

        leftover_tr = tr_f.readline()
        leftover_zh = zh_f.readline()
        if leftover_tr or leftover_zh:
            raise ValueError(
                f"TR-ZH files have mismatched lengths near line {n_seen}: "
                f"{tr_path} vs {zh_path}"
            )

    elapsed = time.time() - t0
    print(
        f"Loaded TR-ZH: {n_seen:,} lines, {len(tr_to_zh):,} unique Turkish "
        f"({n_dup:,} duplicate TR rows) in {elapsed / 60:.1f} min."
    )
    return tr_to_zh


def build_trilingual(
    en_path: Path,
    tr_path: Path,
    tr_to_zh: dict[str, str],
) -> list[dict[str, str]]:
    """Join EN-TR with TR-ZH on Turkish; keep one triple per shared TR.

    Filters during the join:
      - English / Turkish: >= MIN_WORDS words
      - English / Turkish / Chinese: <= MAX_CHARS characters
    """
    triples: list[dict[str, str]] = []
    used_tr: set[str] = set()
    n_seen = 0
    n_tr_hits = 0
    n_filtered = 0
    t0 = time.time()

    print(f"Scanning EN-TR and joining on Turkish:\n  {en_path}\n  {tr_path}")
    with (
        en_path.open("r", encoding="utf-8") as en_f,
        tr_path.open("r", encoding="utf-8") as tr_f,
    ):
        for en_line, tr_line in zip(en_f, tr_f):
            n_seen += 1
            tr = normalize_tr(tr_line)
            if not tr or tr not in tr_to_zh or tr in used_tr:
                if PROGRESS_EVERY > 0 and n_seen % PROGRESS_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = n_seen / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  EN-TR scanned {n_seen:,} | TR hits {n_tr_hits:,} | "
                        f"kept {len(triples):,} | filtered {n_filtered:,} | "
                        f"{rate:,.0f} lines/s",
                        flush=True,
                    )
                continue

            n_tr_hits += 1
            en = en_line.strip()
            zh = tr_to_zh[tr].strip()

            # Length filters (word count on EN/TR; char limit on all sides).
            # Chinese is not space-tokenized reliably, so skip word-count on ZH.
            if (
                word_count(en) < MIN_WORDS
                or word_count(tr) < MIN_WORDS
                or len(en) > MAX_CHARS
                or len(tr) > MAX_CHARS
                or len(zh) > MAX_CHARS
                or not zh
            ):
                n_filtered += 1
                continue

            # Keep one triple per shared Turkish sentence.
            used_tr.add(tr)
            triples.append({"en": en, "tr": tr, "zh": zh})

            if PROGRESS_EVERY > 0 and n_seen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = n_seen / elapsed if elapsed > 0 else 0.0
                print(
                    f"  EN-TR scanned {n_seen:,} | TR hits {n_tr_hits:,} | "
                    f"kept {len(triples):,} | filtered {n_filtered:,} | "
                    f"{rate:,.0f} lines/s",
                    flush=True,
                )

        leftover_en = en_f.readline()
        leftover_tr = tr_f.readline()
        if leftover_en or leftover_tr:
            raise ValueError(
                f"EN-TR files have mismatched lengths near line {n_seen}: "
                f"{en_path} vs {tr_path}"
            )

    elapsed = time.time() - t0
    print(
        f"Finished join: scanned {n_seen:,} EN-TR lines in {elapsed / 60:.1f} min | "
        f"unique shared TR hits {n_tr_hits:,} | "
        f"kept after filter {len(triples):,} | "
        f"dropped by filter {n_filtered:,}."
    )
    return triples


def sample_to_target_en_words(
    dataset: Dataset,
    target_words: int = TARGET_EN_WORDS,
) -> tuple[Dataset, int]:
    """Take shuffled examples until cumulative English words reach target."""
    selected_indices: list[int] = []
    total_en_words = 0

    for i in range(len(dataset)):
        total_en_words += word_count(dataset[i]["en"])
        selected_indices.append(i)
        if total_en_words >= target_words:
            break

    return dataset.select(selected_indices), total_en_words


def extract_10m_trilingual_subset() -> None:
    # 1–2. Load NLLB EN-TR and TR-ZH (TR-ZH as lookup; EN-TR streamed).
    for path in (EN_TR_EN, EN_TR_TR, TR_ZH_TR, TR_ZH_ZH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")

    tr_to_zh = load_tr_to_zh(TR_ZH_TR, TR_ZH_ZH)

    # 3–4. Join on Turkish and filter short / very long sentences.
    triples = build_trilingual(EN_TR_EN, EN_TR_TR, tr_to_zh)
    if not triples:
        raise RuntimeError("No trilingual triples remained after join/filter.")

    dataset = Dataset.from_list(triples)
    print(f"Trilingual dataset size: {len(dataset):,}")

    # 5. Shuffle
    print(f"Shuffling with seed={SEED}...")
    shuffled = dataset.shuffle(seed=SEED)

    # 6. Sample until 10M English words
    print(f"Sampling until {TARGET_EN_WORDS:,} English words...")
    subset, total_en_words = sample_to_target_en_words(shuffled, TARGET_EN_WORDS)
    print(
        f"Selected {len(subset):,} samples with a total of "
        f"{total_en_words:,} English words."
    )
    if total_en_words < TARGET_EN_WORDS:
        print(
            f"WARNING: only reached {total_en_words:,} English words "
            f"(target {TARGET_EN_WORDS:,})."
        )

    # 7. Save language subsets (+ arrow cache like the HPLT script)
    print(f"Saving 10M subset arrow dataset to {ARROW_SAVE_PATH}...")
    ARROW_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(ARROW_SAVE_PATH))

    TXT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    en_path = TXT_SAVE_DIR / "NLLB_10M.en"
    tr_path = TXT_SAVE_DIR / "NLLB_10M.tr"
    zh_path = TXT_SAVE_DIR / "NLLB_10M.zh"

    print(f"Saving separate EN / TR / ZH text files to {TXT_SAVE_DIR}...")
    with (
        en_path.open("w", encoding="utf-8") as f_en,
        tr_path.open("w", encoding="utf-8") as f_tr,
        zh_path.open("w", encoding="utf-8") as f_zh,
    ):
        for item in subset:
            f_en.write(item["en"].replace("\n", " ").strip() + "\n")
            f_tr.write(item["tr"].replace("\n", " ").strip() + "\n")
            f_zh.write(item["zh"].replace("\n", " ").strip() + "\n")

    print("Analyzing the sample dataset (EN-TR stats)...")
    analyze_dataset(subset, "NLLB_10M_subset")
    print("Done.")


if __name__ == "__main__":
    extract_10m_trilingual_subset()
