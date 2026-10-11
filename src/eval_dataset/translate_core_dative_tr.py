"""Translate gold_text and group_span_text from German to English and Turkish.

Turkish: Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-trk
    (opus-mt-tc target prefix ``>>tur<<``).
English: Helsinki-NLP/opus-mt_tiny_deu-eng

Transformers 5 no longer registers pipeline("translation"), so this script
calls MarianMTModel.generate directly.

Example:
    python translate_core_dative_tr.py
    python translate_core_dative_tr.py --limit 5 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Optional, Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(
    _SCRIPT_DIR, "testset", "dative_categories", "core_dative_argument.json"
)
DEFAULT_OUTPUT = os.path.join(
    _SCRIPT_DIR,
    "testset",
    "dative_categories",
    "core_dative_argument_translations.csv",
)
DEFAULT_TR_MODEL = "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-trk"
DEFAULT_EN_MODEL = "Helsinki-NLP/opus-mt_tiny_deu-eng"
DEFAULT_TGT_PREFIX = ">>tur<<"

CSV_FIELD_ORDER = [
    "split",
    "sent_id",
    "group_start",
    "group_end",
    "group_span_text",
    "group_span_text_en",
    "group_span_text_tr",
    "gold_text",
    "gold_text_en",
    "gold_text_tr",
]


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _with_tgt_prefix(text: str, prefix: Optional[str]) -> str:
    if not prefix:
        return text
    stripped = text.lstrip()
    if stripped.startswith(">>") and "<<" in stripped[:16]:
        return text
    return f"{prefix} {text}"


def _load_marian(model_name: str, device: str):
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def _translate_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    tgt_prefix: Optional[str] = None,
) -> List[str]:
    import torch

    device = next(model.parameters()).device
    cleaned = [t if t else "" for t in texts]
    out: List[str] = []
    for i in range(0, len(cleaned), batch_size):
        chunk = cleaned[i : i + batch_size]
        nonempty_idx = [j for j, t in enumerate(chunk) if t.strip()]
        results = [""] * len(chunk)
        if nonempty_idx:
            prefixed = [_with_tgt_prefix(chunk[j], tgt_prefix) for j in nonempty_idx]
            enc = tokenizer(
                prefixed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                generated = model.generate(**enc, max_length=max_length)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for j, pred in zip(nonempty_idx, decoded):
                results[j] = pred
        out.extend(results)
        done = min(i + batch_size, len(cleaned))
        print(f"  translated {done}/{len(cleaned)}", flush=True)
    return out


def _csv_fieldnames(rows: Sequence[dict]) -> List[str]:
    seen = set()
    names: List[str] = []
    for key in CSV_FIELD_ORDER:
        if any(key in row for row in rows):
            names.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def main() -> None:
    p = argparse.ArgumentParser(
        description=
        "Translate gold_text and group_span_text to English and Turkish.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="source JSON list")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="output CSV path")
    p.add_argument(
        "--model",
        default=DEFAULT_TR_MODEL,
        help="Hugging Face German→Turkish model id",
    )
    p.add_argument(
        "--en-model",
        default=DEFAULT_EN_MODEL,
        help="Hugging Face German→English model id",
    )
    p.add_argument(
        "--tgt-prefix",
        default=DEFAULT_TGT_PREFIX,
        help="opus-mt-tc target language token for Turkish, e.g. >>tur<<",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only translate the first N records (for a smoke test)",
    )
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)
    if args.limit is not None:
        records = records[:args.limit]

    device = _device()
    golds = [r.get("gold_text", "") for r in records]
    spans = [r.get("group_span_text", "") for r in records]

    print(f"Loading {args.en_model} on {device} ...", flush=True)
    en_model, en_tok = _load_marian(args.en_model, device)
    print(f"Translating {len(golds)} gold_text fields → EN ...", flush=True)
    gold_en = _translate_batch(en_model, en_tok, golds, args.batch_size,
                               args.max_length)
    print(f"Translating {len(spans)} group_span_text fields → EN ...",
          flush=True)
    span_en = _translate_batch(en_model, en_tok, spans, args.batch_size,
                               args.max_length)
    del en_model, en_tok

    print(f"Loading {args.model} on {device} ...", flush=True)
    tr_model, tr_tok = _load_marian(args.model, device)
    print(f"Translating {len(golds)} gold_text fields → TR ...", flush=True)
    gold_tr = _translate_batch(tr_model, tr_tok, golds, args.batch_size,
                               args.max_length, args.tgt_prefix)
    print(f"Translating {len(spans)} group_span_text fields → TR ...",
          flush=True)
    span_tr = _translate_batch(tr_model, tr_tok, spans, args.batch_size,
                               args.max_length, args.tgt_prefix)
    del tr_model, tr_tok

    out_records = []
    for rec, g_en, g_tr, s_en, s_tr in zip(records, gold_en, gold_tr, span_en,
                                           span_tr):
        row = dict(rec)
        row["gold_text_en"] = g_en
        row["gold_text_tr"] = g_tr
        row["group_span_text_en"] = s_en
        row["group_span_text_tr"] = s_tr
        row.pop("corrupted")
        row.pop("category")
        row.pop("dative_type")
        out_records.append(row)

    fieldnames = _csv_fieldnames(out_records)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_records)
    print(f"Wrote {len(out_records)} records to {args.output}")


if __name__ == "__main__":
    main()
