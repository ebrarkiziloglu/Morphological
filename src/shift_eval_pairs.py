#!/usr/bin/env python3
"""
Shift text fields in the accusative/dative pair testsets by +1 (ord(c) + 1).

Applies the same letter/digit shifting as ``shift_chars.py`` to
``group_span_text``, ``gold_text``, and ``corrupted``. Writes
``*_shifted.json`` copies alongside the originals.

Usage:
    python src/shift_eval_pairs.py
    python src/shift_eval_pairs.py --inputs path/a.json path/b.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shift_chars import make_shifted_path, shift_json_file  # noqa: E402

DEFAULT_INPUTS = [
    os.path.expanduser(
        "~/Morphological/src/eval_dataset/accusative_dative_pairs.json"
    ),
    os.path.expanduser(
        "~/Morphological/src/eval_dataset/dative_accusative_pairs.json"
    ),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Shift chars by +1 in accusative/dative pair testsets."
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="JSON pair files to shift (default: both pair testsets).",
    )
    p.add_argument(
        "--suffix",
        default="_shifted",
        help="Suffix appended to output filenames (default: '_shifted').",
    )
    return p.parse_args()


def main():
    args = parse_args()

    for src in args.inputs:
        dst = make_shifted_path(src, args.suffix)
        shift_json_file(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
