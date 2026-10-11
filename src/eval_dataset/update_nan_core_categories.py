"""Replace NaN_core categories in the dative-accusative pairs test set.

For each item in dative_accusative_pairs.json whose category is NaN_core,
look up the matching item in core_dative_argument_new.json (same split,
sent_id, group_start, group_end, group_span_text) and set category to that
item's final_category.

Example:
    python update_nan_core_categories.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAIRS = os.path.join(
    _SCRIPT_DIR, "testset", "dative_accusative_pairs.json"
)
DEFAULT_CORE = os.path.join(
    _SCRIPT_DIR, "testset", "dative_categories", "core_dative_argument_new.json"
)

KEY_FIELDS = ("split", "sent_id", "group_start", "group_end", "group_span_text")
SOURCE_CATEGORY = "NaN_core"


def _item_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def load_final_categories(path: str) -> Dict[Tuple[str, str, str, str, str], str]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    categories: Dict[Tuple[str, str, str, str, str], str] = {}
    for item in items:
        key = _item_key(item)
        if key in categories:
            raise ValueError(f"Duplicate core JSON key: {key}")
        if "final_category" not in item:
            raise ValueError(f"Missing final_category for key: {key}")
        categories[key] = item["final_category"]
    return categories


def update_nan_core_categories(
    pairs: List[Dict[str, Any]],
    final_categories: Dict[Tuple[str, str, str, str, str], str],
) -> Tuple[int, int]:
    updated = 0
    unmatched = 0
    for item in pairs:
        if item.get("category") != SOURCE_CATEGORY:
            continue
        key = _item_key(item)
        final_category = final_categories.get(key)
        if final_category is None:
            unmatched += 1
            continue
        item["category"] = final_category
        updated += 1
    return updated, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace NaN_core category values in dative_accusative_pairs.json "
            "with final_category from core_dative_argument_new.json."
        )
    )
    parser.add_argument("--pairs", default=DEFAULT_PAIRS, help="Pairs JSON path")
    parser.add_argument(
        "--core", default=DEFAULT_CORE, help="Updated core dative JSON path"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (defaults to overwriting --pairs)",
    )
    args = parser.parse_args()
    output_path = args.output or args.pairs

    with open(args.pairs, encoding="utf-8") as f:
        pairs = json.load(f)

    final_categories = load_final_categories(args.core)
    nan_core_count = sum(1 for item in pairs if item.get("category") == SOURCE_CATEGORY)
    updated, unmatched = update_nan_core_categories(pairs, final_categories)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
        f.write("\n")

    remaining = sum(1 for item in pairs if item.get("category") == SOURCE_CATEGORY)
    print(f"Pairs items:           {len(pairs)}")
    print(f"NaN_core before:       {nan_core_count}")
    print(f"Updated from core:     {updated}")
    print(f"Unmatched NaN_core:    {unmatched}")
    print(f"NaN_core remaining:    {remaining}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
