#!/usr/bin/env python3
"""POS-tag the NLLB Turkish corpus with ``postagger.PosTagger``.

Reads one sentence per line and writes JSONL records:

    {"text": "...", "tokens": [...], "tag_list": [...]}

``PosTagger`` was trained as ``Buseak/pos_tagger_3112_v3`` and loads both
the model and its tokenizer from that Hugging Face repo.

Usage (from the Morphological repo root)::

    python src/pretraining_dataset/postag_NLLB_tr.py

    python src/pretraining_dataset/postag_NLLB_tr.py \\
        --input data/NLLB_10M_subset/NLLB_10M.tr \\
        --output data/NLLB_10M_subset/NLLB_10M.tr.pos.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HF_MODEL_NAME = "Buseak/pos_tagger_3112_v3"

DEFAULT_INPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_small.tr"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_small.tr.pos.jsonl"

sys.path.insert(0, str(SCRIPT_DIR))


def _move_tokenizer_outputs_to_model_device(tagger) -> None:
    """Keep tokenizer tensors on the same device as the tagger weights."""
    original_call = tagger.tokenizer.__call__

    def __call__(*args, **kwargs):
        encoded = original_call(*args, **kwargs)
        if kwargs.get("return_tensors") != "pt":
            return encoded
        device = next(tagger.model.parameters()).device
        return encoded.to(device)

    tagger.tokenizer.__call__ = __call__


def load_tagger(device: str):
    import torch

    from postagger import PosTagger

    print(f"Loading POS tagger ({HF_MODEL_NAME}) on {device}...", flush=True)
    tagger = PosTagger()
    tagger.model.to(device)
    tagger.model.eval()
    _move_tokenizer_outputs_to_model_device(tagger)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return tagger


def tagged_line(tokens: list[str], tags: list[str]) -> str:
    n = min(len(tokens), len(tags))
    return " ".join(f"{tokens[i]}/{tags[i]}" for i in range(n))


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def postag_file(
    input_path: Path,
    output_path: Path,
    device: str,
    max_lines: int | None = None,
    tagged_output: Path | None = None,
) -> None:
    from tqdm import tqdm

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    tagger = load_tagger(device)
    n_lines = count_lines(input_path)
    if max_lines is not None:
        n_lines = min(n_lines, max_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tagged_handle = None
    if tagged_output is not None:
        tagged_output.parent.mkdir(parents=True, exist_ok=True)
        tagged_handle = tagged_output.open("w", encoding="utf-8")

    n_ok = 0
    n_empty = 0
    n_fail = 0
    t0 = time.time()

    try:
        with (
            input_path.open("r", encoding="utf-8") as in_f,
            output_path.open("w", encoding="utf-8") as out_f,
        ):
            for i, raw_line in enumerate(
                tqdm(in_f, total=n_lines, desc="POS-tagging", unit="sent")
            ):
                if max_lines is not None and i >= max_lines:
                    break

                text = raw_line.rstrip("\n")
                record = {"text": text, "tokens": [], "tag_list": []}

                if not text.strip():
                    n_empty += 1
                else:
                    try:
                        result = tagger.pos_tag(text)
                        record["tokens"] = result["tokens"]
                        record["tag_list"] = result["tag_list"]
                        n_ok += 1
                    except Exception as exc:  # noqa: BLE001 — keep the run going
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        n_fail += 1

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if tagged_handle is not None:
                    tagged_handle.write(
                        tagged_line(record["tokens"], record["tag_list"]) + "\n"
                    )
    finally:
        if tagged_handle is not None:
            tagged_handle.close()

    elapsed = time.time() - t0
    rate = (n_ok + n_empty + n_fail) / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {elapsed:.1f}s ({rate:.1f} sent/s). "
        f"tagged={n_ok:,} empty={n_empty:,} failed={n_fail:,}",
        flush=True,
    )
    print(f"JSONL written to {output_path}", flush=True)
    if tagged_output is not None:
        print(f"Tagged text written to {tagged_output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POS-tag NLLB Turkish sentences with postagger.PosTagger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="One sentence per line")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL output path")
    parser.add_argument(
        "--tagged-output",
        type=Path,
        default=None,
        help="Optional token/TAG text file (one sentence per line)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if _cuda_available() else "cpu",
        help="torch device for the tagger",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Stop after this many input lines (useful for smoke tests)",
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    args = parse_args()
    postag_file(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        max_lines=args.max_lines,
        tagged_output=args.tagged_output,
    )


if __name__ == "__main__":
    main()
