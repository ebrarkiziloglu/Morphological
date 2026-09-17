#!/usr/bin/env python3
"""
Shift every character in the German datasets by +1 (ord(c) + 1).

Produces *_shifted.* copies alongside the originals so nothing is overwritten.

Usage:
    python src/shift_chars.py                          # uses defaults below
    python src/shift_chars.py --pretrain_path ...      # override paths
"""

import argparse
import glob
import json
import os
import sys


def shift_text(text: str) -> str:
    """Return *text* with every letter or digit shifted by +1.

    Letters (including German umlauts ä/ö/ü/ß and other Unicode letters via
    ``str.isalpha()``) and digits (``str.isdigit()``) are shifted. Whitespace,
    newlines and punctuation are left untouched so that line-by-line dataset
    readers, tokenizer pre-tokenisation and structural markers (e.g. JSON
    syntax, sentence punctuation) remain intact.
    Unicode codepoint overflow (chr(0x10FFFF)+1) is guarded by clamping.
    """
    out = []
    for c in text:
        if c.isalpha() or c.isdigit():
            code = ord(c) + 1
            if code > 0x10FFFF:
                code = 0x10FFFF
            out.append(chr(code))
        else:
            out.append(c)
    return "".join(out)


# ── Plain-text pretraining data ──────────────────────────────────────────────

def shift_pretrain_file(src_path: str, dst_path: str) -> None:
    """Read a plain-text file, shift every char by +1, write to *dst_path*."""
    if not os.path.isfile(src_path):
        print(f"[ERROR] Pretraining file not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    with open(src_path, "r", encoding="utf-8") as fin:
        original = fin.read()

    shifted = shift_text(original)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as fout:
        fout.write(shifted)

    print(f"[Pretrain] {src_path}  →  {dst_path}  ({len(original)} chars shifted)")


# ── JSON eval data ───────────────────────────────────────────────────────────

TEXT_FIELDS = ("gold_text", "corrupted", "group_span_text")


def shift_json_file(src_path: str, dst_path: str) -> None:
    """Load a JSON array, shift `gold_text` and `corrupted` fields, save."""
    if not os.path.isfile(src_path):
        print(f"[ERROR] JSON file not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    with open(src_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    count = 0
    for entry in data:
        for field in TEXT_FIELDS:
            if field in entry:
                entry[field] = shift_text(entry[field])
                count += 1

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)

    print(f"[Eval] {src_path}  →  {dst_path}  ({count} fields shifted)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Shift chars by +1 in German datasets.")
    p.add_argument(
        "--pretrain_path",
        default=os.path.expanduser(
            "~/Morphological/data/BabyLM_DE_subset/BabyLM_DE_subset.de"
        ),
        help="Path to plain-text pretraining file.",
    )
    p.add_argument(
        "--eval_dir",
        default=os.path.expanduser(
            "~/Morphological/src/eval_dataset/"
        ),
        help="Directory containing JSON eval files (searched recursively).",
    )
    p.add_argument(
        "--suffix",
        default="_shifted",
        help="Suffix appended to output filenames (default: '_shifted').",
    )
    return p.parse_args()


def make_shifted_path(path: str, suffix: str) -> str:
    """Insert *suffix* before the file extension.
    e.g. foo.de → foo_shifted.de,  bar.json → bar_shifted.json"""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def main():
    args = parse_args()

    # 1. Pretraining plain-text file
    if args.pretrain_path:
        dst_pretrain = make_shifted_path(args.pretrain_path, args.suffix)
        shift_pretrain_file(args.pretrain_path, dst_pretrain)

    # 2. All JSON files under the eval directory
    json_pattern = os.path.join(args.eval_dir, "**", "*.json")
    json_files = sorted(glob.glob(json_pattern, recursive=True))

    if not json_files:
        print(f"[WARN] No JSON files found under {args.eval_dir}", file=sys.stderr)

    for jf in json_files:
        dst_json = make_shifted_path(jf, args.suffix)
        shift_json_file(jf, dst_json)

    print("\nDone.")


if __name__ == "__main__":
    main()
