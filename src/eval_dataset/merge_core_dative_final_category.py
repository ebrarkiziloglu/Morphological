"""Merge final_category from the cleaned CSV into core_dative_argument.json.

Items are matched on split, sent_id, group_start, group_end, and group_span_text.

Example:
    python merge_core_dative_final_category.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(
    _SCRIPT_DIR, "testset", "dative_categories", "core_dative_argument.json"
)
DEFAULT_CSV = os.path.join(
    _SCRIPT_DIR, "testset", "core_dative_argument cleaned.csv"
)
DEFAULT_OUTPUT = os.path.join(
    _SCRIPT_DIR, "testset", "dative_categories", "core_dative_argument_new.json"
)

KEY_FIELDS = ("split", "sent_id", "group_start", "group_end", "group_span_text")


def _item_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def load_csv_categories(path: str) -> Dict[Tuple[str, str, str, str, str], str]:
    categories: Dict[Tuple[str, str, str, str, str], str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            key = _item_key(row)
            if key in categories:
                raise ValueError(f"Duplicate CSV key: {key}")
            categories[key] = row["final_category"]
    return categories


def merge_final_category(
    json_items: List[Dict[str, Any]],
    categories: Dict[Tuple[str, str, str, str, str], str],
) -> Tuple[List[Dict[str, Any]], int]:
    matched = 0
    updated: List[Dict[str, Any]] = []
    for item in json_items:
        new_item = dict(item)
        key = _item_key(item)
        if key in categories:
            new_item["final_category"] = categories[key]
            matched += 1
        updated.append(new_item)
    return updated, matched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add CSV final_category to matching core_dative_argument JSON items."
    )
    parser.add_argument("--json", default=DEFAULT_JSON, help="Input JSON path")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        json_items = json.load(f)

    categories = load_csv_categories(args.csv)
    updated, matched = merge_final_category(json_items, categories)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
        f.write("\n")

    unmatched_json = len(json_items) - matched
    unmatched_csv = len(categories) - matched
    print(f"JSON items: {len(json_items)}")
    print(f"CSV items:  {len(categories)}")
    print(f"Matched:    {matched}")
    print(f"Unmatched JSON items: {unmatched_json}")
    print(f"Unmatched CSV items:  {unmatched_csv}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
