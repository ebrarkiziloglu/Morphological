#!/usr/bin/env python3
"""
Evaluate a case-stripped Turkish file against its original.

The input and stripped files must be line- and whitespace-token-aligned (as
written by ``remove_cases_from_tr_nllb.py``). Each original sentence is analysed
with Zemberek and disambiguated in context; that disambiguated analysis is the
reference used below.

Reported
--------
- missed:          reference analysis has a case / -ki, but the token is unchanged
- false strips:    token was changed, but the reference analysis has no case / -ki
                   (e.g. para -> par, Türkiye -> Türki)
- residual:        stripped token whose Zemberek analyses *all* still carry a case / -ki
- unknown output:  stripped token Zemberek cannot analyse at all (non-words)
- every changed token, with its reference analysis

Usage (from the Morphological repo root)::

    python src/pretraining_dataset/remove_cases_from_turkish/evaluate_stripping.py \\
        --original data/NLLB_10M_subset/NLLB_small.tr \\
        --stripped data/NLLB_10M_subset/NLLB_small_stripped.tr \\
        --log src/pretraining_dataset/remove_cases_from_turkish/logs/eval.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import remove_case_suffixes_from_tr_zemberek as zemb  # noqa: E402

# Every Zemberek case (incl. instrumental, equative) + relative -ki
CASE_OR_REL = zemb.TARGET_CASES | {"Ins", "Equ", "Rel"}
APOSTROPHES = "'’‘"


def peel_punct(token: str) -> str:
    """Core of a token without leading/trailing punctuation (same rule as the pipeline)."""

    def is_core_char(char: str) -> bool:
        return char.isalnum() or char in APOSTROPHES or char == "%"

    start, end = 0, len(token)
    while start < end and not is_core_char(token[start]):
        start += 1
    while end > start and not is_core_char(token[end - 1]):
        end -= 1
    return token[start:end]


def has_case(parse) -> bool:
    return any(tag in CASE_OR_REL for tag in zemb.morpheme_ids(parse))


def reference_parses(cores: list[str]) -> list:
    """Context-disambiguated Zemberek analysis per core (None if unknown/empty)."""
    analyses = [zemb.morphology.analyze(core or ".") for core in cores]
    try:
        best = zemb.morphology.disambiguate(" ".join(cores), analyses).best_analysis()
    except Exception:  # noqa: BLE001
        best = [None] * len(cores)
    if len(best) != len(cores):
        best = [None] * len(cores)
    out = []
    for core, parse in zip(cores, best):
        if not core or parse is None or parse.is_unknown():
            out.append(None)
        else:
            out.append(parse)
    return out


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def evaluate(original: Path, stripped: Path, title: str) -> str:
    n_tokens = n_changed = n_ref_case = 0
    missed: Counter = Counter()
    false_strips: Counter = Counter()
    residual: Counter = Counter()
    unknown_out: Counter = Counter()
    changes: Counter = Counter()
    misaligned = 0

    with original.open(encoding="utf-8") as f_in, stripped.open(encoding="utf-8") as f_out:
        for line_no, (a, b) in enumerate(zip(f_in, f_out), 1):
            src, out = a.split(), b.split()
            if len(src) != len(out):
                misaligned += 1
                continue
            src_cores = [peel_punct(t) for t in src]
            refs = reference_parses(src_cores)
            for tok_in, tok_out, core, ref in zip(src, out, src_cores, refs):
                if not core:
                    continue
                n_tokens += 1
                changed = tok_in != tok_out
                ref_str = ref.format_string() if ref is not None else "<unknown>"
                ref_case = ref is not None and has_case(ref)
                n_ref_case += ref_case
                out_core = peel_punct(tok_out)
                if changed:
                    n_changed += 1
                    changes[(core, out_core, ref_str)] += 1
                    if ref is not None and not ref_case:
                        false_strips[(core, out_core, ref_str)] += 1
                    out_parses = [
                        p for p in zemb.morphology.analyze(out_core) if not p.is_unknown()
                    ]
                    if not out_parses:
                        unknown_out[(core, out_core)] += 1
                    elif all(has_case(p) for p in out_parses):
                        residual[(core, out_core, out_parses[0].format_string())] += 1
                elif ref_case:
                    missed[(core, ref_str)] += 1

    def pct(x: int, y: int) -> str:
        return f"{100.0 * x / y:.1f}%" if y else "0.0%"

    n_missed = sum(missed.values())
    n_false = sum(false_strips.values())
    lines = [
        f"# {title}",
        f"date:      {datetime.now().isoformat(timespec='seconds')}",
        f"git HEAD:  {git_rev()} (+ uncommitted changes, if any)",
        f"original:  {original}",
        f"stripped:  {stripped}",
        "",
        "Reference = Zemberek analysis disambiguated in sentence context (no POS tagger).",
        "",
        "Summary",
        "-------",
        f"tokens (non-punct):            {n_tokens:,}",
        f"changed tokens:                {n_changed:,} ({pct(n_changed, n_tokens)})",
        f"reference has case/-ki:        {n_ref_case:,}",
        f"missed (case, unchanged):      {n_missed:,} ({pct(n_missed, n_ref_case)} of case tokens)",
        f"false strips (no case, changed): {n_false:,} ({pct(n_false, n_changed)} of changed)",
        f"residual case in output:       {sum(residual.values()):,}",
        f"unknown (non-word) output:     {sum(unknown_out.values()):,}",
        f"misaligned lines (skipped):    {misaligned:,}",
        "",
    ]

    def section(name: str, counter: Counter, fmt) -> None:
        lines.append(f"{name} ({sum(counter.values()):,} tokens, {len(counter):,} types)")
        lines.append("-" * len(lines[-1]))
        for key, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"{count:4d}  {fmt(key)}")
        lines.append("")

    section("Missed", missed, lambda k: f"{k[0]}\t{k[1]}")
    section("False strips", false_strips, lambda k: f"{k[0]} -> {k[1]}\t{k[2]}")
    section("Residual case in output", residual, lambda k: f"{k[0]} -> {k[1]}\t{k[2]}")
    section("Unknown output", unknown_out, lambda k: f"{k[0]} -> {k[1]}")
    section("All changes", changes, lambda k: f"{k[0]} -> {k[1]}\t{k[2]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--stripped", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--title", default="Case-stripping evaluation")
    args = parser.parse_args()

    report = evaluate(args.original, args.stripped, args.title)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(report + "\n", encoding="utf-8")
    print(report.split("\nMissed")[0])
    print(f"Wrote {args.log}")


if __name__ == "__main__":
    main()
