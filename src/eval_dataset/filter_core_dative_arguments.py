#!/usr/bin/env python3
"""Keep only core_dative_argument examples from a dative/accusative pair file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "dative_accusative_pairs_v0.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "dative_accusative_pairs_v0_core.json"
KEEP_TYPE = "core_dative_argument"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Filter pair JSON to dative_type={KEEP_TYPE!r}",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = json.loads(args.input.read_text(encoding="utf-8"))
    kept = [ex for ex in examples if ex.get("dative_type") == KEEP_TYPE]
    args.output.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Kept {len(kept):,} / {len(examples):,} examples -> {args.output}")


if __name__ == "__main__":
    main()
