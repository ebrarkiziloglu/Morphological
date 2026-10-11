"""
Create case-corrupted minimal pairs from UD German-GSD (NP-span level).

``--direction acc2dat`` (default): rewrite every ``Case=Acc`` token of an
accusative NP span to its dative form.
``--direction dat2acc``: rewrite every ``Case=Dat`` token of a dative NP span to
its accusative form, attach a ``dative_type`` label from
``dative_spans/german_datives_all.csv`` and write one JSON per label under
``testset/dative_categories/``.

Surface forms come only from ``german_ud_lookup_dictionary.csv`` (matching
Lemma, Number, Gender, Upos; for Plur, any Gender is used only when no
same-Gender row exists). Spans with a token that has no matching row are
skipped and logged in ``*_unconverted.json``; other spans in the same sentence
are still written. Rows where the corrupt sentence equals ``gold_text`` are
omitted and the output holds one row per NP span.

acc2dat only: a NOUN with no matching dictionary row is left unconverted
instead of dropping the span, as long as another token in it (determiner /
adjective) could be corrupted.
dat2acc only: a NOUN/PROPN whose tagged gender has no Acc row is still
converted when the gold surface is already an Acc form of that lemma.

Pairs whose NP span touches a dictionary cell with several forms are also
written to a separate duplicates JSON (``gold_span_duplicate_forms``).

An elimination summary (counts + reasons) is printed at the end.

Example:
    python corrupt_pairs.py --direction acc2dat
    python corrupt_pairs.py --direction dat2acc
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Set

from load_ud_dataset import UD_GERMAN_GSD_SPLITS
from ud_corruption_utils import run_pair_generation_main

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "data"))

SPLIT_ORDER = ("train", "test", "dev")

# direction -> (source_case, target_case, label, file stem)
DIRECTIONS = {
    "acc2dat": ("Acc", "Dat", "Acc→Dat", "accusative_dative"),
    "dat2acc": ("Dat", "Acc", "Dat→Acc", "dative_accusative"),
}


def _allow_missing_for_upos(split: str) -> Optional[Set[str]]:
    """acc2dat: tolerate unconverted NOUNs on every split (see module docstring)."""
    _ = split
    return {"NOUN"}


def main(
    direction: str,
    ud_base_path: str,
    output_group_json: str,
    dict_csv_path: str,
    log_level: str = "INFO",
    dative_types_csv_path: Optional[str] = None,
    duplicates_dict_csv_path: Optional[str] = None,
    output_group_duplicates_json: Optional[str] = None,
    output_dative_categories_dir: Optional[str] = None,
):
    """Generate pairs for ``direction`` (``acc2dat`` or ``dat2acc``)."""
    source_case, target_case, label, stem = DIRECTIONS[direction]
    kwargs = {}
    if direction == "acc2dat":
        kwargs["allow_missing_for_upos_fn"] = _allow_missing_for_upos
    else:
        kwargs["dative_types_csv_path"] = dative_types_csv_path or os.path.join(
            _SCRIPT_DIR, "dative_spans", "german_datives_all.csv"
        )
        kwargs["output_dative_categories_dir"] = output_dative_categories_dir

    return run_pair_generation_main(
        ud_base_path=ud_base_path,
        splits={k: UD_GERMAN_GSD_SPLITS[k] for k in SPLIT_ORDER},
        output_group_json=output_group_json,
        dict_csv_path=dict_csv_path,
        source_case=source_case,
        target_case=target_case,
        direction_label=label,
        log_level=log_level,
        output_unconverted_json=os.path.join(
            os.path.dirname(output_group_json), f"{stem}_unconverted.json"
        ),
        duplicates_dict_csv_path=duplicates_dict_csv_path,
        output_group_duplicates_json=output_group_duplicates_json,
        **kwargs,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Acc↔Dat minimal-pair export with elimination diagnostics."
    )
    p.add_argument(
        "--direction",
        choices=sorted(DIRECTIONS),
        default="acc2dat",
        help="acc2dat: Acc→Dat pairs; dat2acc: Dat→Acc pairs (default acc2dat)",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("ACC_DAT_PAIR_LOG_LEVEL", "INFO"),
        help="logging level for stderr (default INFO; try DEBUG)",
    )
    p.add_argument("--ud-base", default=None, help="directory with UD_German-GSD CoNLL-U files")
    p.add_argument("--dictionary", default=None, help="path to german_ud_lookup_dictionary.csv")
    p.add_argument("--output-group-json", default=None, help="output JSON path")
    p.add_argument(
        "--dative-types-csv",
        default=None,
        help="dat2acc only: dative_spans/german_datives_all.csv for dative_type labels",
    )
    p.add_argument(
        "--duplicates-dictionary",
        default=None,
        help="duplicates/german_ud_lookup_dictionary_duplicates.csv",
    )
    p.add_argument(
        "--output-group-duplicates-json",
        default=None,
        help="JSON for pairs containing duplicate-dictionary surface forms",
    )
    p.add_argument(
        "--output-dative-categories-dir",
        default=None,
        help="dat2acc only: directory for per-dative_type JSONs (default testset/dative_categories)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stem = DIRECTIONS[args.direction][3]
    dat2acc = args.direction == "dat2acc"
    main(
        direction=args.direction,
        ud_base_path=args.ud_base or os.path.join(_DATA_ROOT, "UD_German-GSD"),
        output_group_json=args.output_group_json
        or os.path.join(_SCRIPT_DIR, "testset", f"{stem}_pairs.json"),
        dict_csv_path=args.dictionary
        or os.path.join(_SCRIPT_DIR, "german_ud_lookup_dictionary.csv"),
        log_level=str(args.log_level).upper(),
        dative_types_csv_path=args.dative_types_csv,
        duplicates_dict_csv_path=args.duplicates_dictionary
        or os.path.join(
            _SCRIPT_DIR, "duplicates", "german_ud_lookup_dictionary_duplicates.csv"
        ),
        output_group_duplicates_json=args.output_group_duplicates_json
        or os.path.join(_SCRIPT_DIR, "duplicates", f"{stem}_pairs_duplicates.json"),
        output_dative_categories_dir=(
            args.output_dative_categories_dir
            or os.path.join(_SCRIPT_DIR, "testset", "dative_categories")
            if dat2acc
            else None
        ),
    )
