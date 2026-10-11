"""Label every dative NP head in UD German-GSD with a ``dative_type``.

Writes ``dative_spans/german_datives_all.csv``, which ``corrupt_pairs.py
--direction dat2acc`` reads to type each Dat→Acc pair.

Example:
    python german_ud_dative_types.py
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from conllu import parse_incr

from load_ud_dataset import UD_GERMAN_GSD_SPLITS
from ud_corruption_utils import (
    case_marker_lemma_for_head_id,
    classify_dative_type,
    is_target_dative_preposition,
    normalize_case_prep_lemma,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UD_BASE = os.path.join(_SCRIPT_DIR, "..", "..", "data", "UD_German-GSD")
DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "dative_spans", "german_datives_all.csv")

DAT_POS = {"NOUN", "PROPN", "PRON"}
SPAN_DEPRELS = {"det", "amod", "nummod", "compound", "fixed", "flat"}


def collect_span(sent, head_id):
    """Simple NP span around the dative head (no preposition / case marker)."""
    return [
        tok
        for tok in sent
        if tok["id"] == head_id
        or (tok["head"] == head_id and tok["deprel"] in SPAN_DEPRELS)
    ]


def get_preposition(case_lemma):
    """Target preposition (aus/bei/…/in); None for nordwestlich, dank, …"""
    if not is_target_dative_preposition(case_lemma):
        return None
    return normalize_case_prep_lemma(case_lemma)


def dative_rows(sent, split):
    """One row per dative NOUN/PROPN/PRON token in ``sent``."""
    by_id = {tok["id"]: tok for tok in sent}
    for tok_idx, tok in enumerate(sent):
        if tok["upos"] not in DAT_POS or (tok["feats"] or {}).get("Case") != "Dat":
            continue
        case_marker = case_marker_lemma_for_head_id(sent, tok["id"])
        gov = by_id.get(tok["head"])
        yield {
            "sent_id": sent.metadata.get("sent_id", ""),
            "sentence": sent.metadata.get("text", ""),
            "span": " ".join(t["form"] for t in collect_span(sent, tok["id"])),
            "head_form": tok["form"],
            "head_lemma": tok["lemma"],
            "head_upos": tok["upos"],
            "case_marker": case_marker,
            "preposition": get_preposition(case_marker),
            "dative_type": classify_dative_type(
                tok,
                case_marker,
                tokens=sent,
                head_idx=tok_idx,
                span_start=tok_idx,
                span_end=tok_idx,
            ),
            "relation": tok["deprel"],
            "governor": gov["form"] if gov else "ROOT",
            "governor_upos": gov["upos"] if gov else "ROOT",
            "split": split,
        }


def main(ud_base: str, output_csv: str) -> pd.DataFrame:
    rows = []
    for split, filename in UD_GERMAN_GSD_SPLITS.items():
        with open(os.path.join(ud_base, filename), encoding="utf-8") as f:
            for sent in parse_incr(f):
                rows.extend(dative_rows(sent, split))

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(df.head())
    print(f"Dative types: {set(df.dative_type)}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ud-base", default=DEFAULT_UD_BASE, help="UD_German-GSD directory")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="output CSV path")
    args = p.parse_args()
    main(args.ud_base, args.output)
