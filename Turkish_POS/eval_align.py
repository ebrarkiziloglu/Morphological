"""Span-align POS tags when gold and predicted tokenizations differ."""
from __future__ import annotations

import ast
import csv
from pathlib import Path


def locate_tokens(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Greedy left-to-right character spans for tokens inside text.

    TULAP gold tokens are often stems (prefixes of surface words), so we
    accept an exact match at the cursor, then a forward search.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    n = len(text)
    for tok in tokens:
        if tok == "":
            spans.append((cursor, cursor))
            continue
        while cursor < n and text[cursor].isspace():
            cursor += 1
        start = -1
        if text.startswith(tok, cursor):
            start = cursor
        else:
            start = text.find(tok, cursor)
        if start == -1:
            raise ValueError(
                f"Could not locate token {tok!r} in remaining text {text[cursor:]!r}"
            )
        end = start + len(tok)
        spans.append((start, end))
        cursor = end
    return spans


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def project_tags(
    source_spans: list[tuple[int, int]],
    source_tags: list[str],
    target_spans: list[tuple[int, int]],
) -> list[str | None]:
    """For each target span, take the source tag with maximum character overlap.

    Ties are broken by the longer source token (the stem/head, not a clitic).
    """
    projected: list[str | None] = []
    for tspan in target_spans:
        best_tag: str | None = None
        best_ov = 0
        best_len = -1
        for sspan, stag in zip(source_spans, source_tags):
            ov = _overlap(tspan, sspan)
            slen = sspan[1] - sspan[0]
            if ov > best_ov or (ov == best_ov and ov > 0 and slen > best_len):
                best_ov = ov
                best_len = slen
                best_tag = stag
        projected.append(best_tag)
    return projected


def aligned_accuracy(
    text: str,
    gold_tokens: list[str],
    gold_tags: list[str],
    pred_tokens: list[str],
    pred_tags: list[str],
    tag_map: dict[str, str] | None = None,
) -> dict:
    gold_spans = locate_tokens(text, gold_tokens)
    pred_spans = locate_tokens(text, pred_tokens)

    if tag_map:
        pred_tags = [tag_map.get(t, t) for t in pred_tags]

    # Precision-oriented: gold tag projected onto each predicted token
    gold_on_pred = project_tags(gold_spans, gold_tags, pred_spans)
    pred_correct = sum(
        p == g for p, g in zip(pred_tags, gold_on_pred) if g is not None
    )
    pred_total = sum(g is not None for g in gold_on_pred)

    # Recall-oriented: predicted tag projected onto each gold token
    pred_on_gold = project_tags(pred_spans, pred_tags, gold_spans)
    gold_correct = sum(
        p == g for p, g in zip(pred_on_gold, gold_tags) if p is not None
    )
    gold_total = sum(p is not None for p in pred_on_gold)

    precision = pred_correct / pred_total if pred_total else 0.0
    recall = gold_correct / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_correct": pred_correct,
        "pred_total": pred_total,
        "gold_correct": gold_correct,
        "gold_total": gold_total,
        "gold_on_pred": gold_on_pred,
        "pred_on_gold": pred_on_gold,
        "pred_tokens": pred_tokens,
        "pred_tags": pred_tags,
        "gold_tokens": gold_tokens,
        "gold_tags": gold_tags,
    }


def evaluate_predictions(gold_rows, predictions, show_mismatches=True):
    """Evaluate a list of (token, tag) sequences against gold rows.

    Returns micro-averaged precision, recall, F1, and word-level accuracy.
    Word-level accuracy = precision: gold tags projected onto predicted tokens.
    """
    pred_correct = pred_total = gold_correct = gold_total = 0
    per_sentence = []
    for i, (row, pred) in enumerate(zip(gold_rows, predictions)):
        pred_tokens = [t[0] for t in pred]
        pred_tags = [t[1] for t in pred]
        r = aligned_accuracy(
            row["text"], row["tokens"], row["tags"], pred_tokens, pred_tags
        )
        pred_correct += r["pred_correct"]
        pred_total += r["pred_total"]
        gold_correct += r["gold_correct"]
        gold_total += r["gold_total"]
        per_sentence.append(r)
        if show_mismatches:
            mismatches = [
                (tok, p, g)
                for tok, p, g in zip(r["pred_tokens"], r["pred_tags"], r["gold_on_pred"])
                if p != g
            ]
            print(
                f"s{i:02d}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"F1={r['f1']:.3f}  gold_n={len(row['tags'])}  "
                f"pred_n={len(pred_tags)}  mismatches={len(mismatches)}"
            )
            for tok, p, g in mismatches:
                print(f"      {tok!r:25s}  pred={p:8s}  gold~={g}")

    precision = pred_correct / pred_total if pred_total else 0.0
    recall = gold_correct / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    print()
    print(
        f"Word-level accuracy (gold tags projected onto predicted tokens): "
        f"{precision:.4f}  ({pred_correct}/{pred_total})"
    )
    print(
        f"Gold-token recall   (predicted tags projected onto gold tokens): "
        f"{recall:.4f}  ({gold_correct}/{gold_total})"
    )
    print(f"Span-aligned F1: {f1:.4f}")
    return {
        "accuracy": precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_correct": pred_correct,
        "pred_total": pred_total,
        "gold_correct": gold_correct,
        "gold_total": gold_total,
        "per_sentence": per_sentence,
    }


def load_gold(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(
                {
                    "text": row["text"],
                    "tokens": ast.literal_eval(row["tokens"]),
                    "tags": ast.literal_eval(row["tags"]),
                }
            )
    return rows


if __name__ == "__main__":
    gold = load_gold(str(Path(__file__).parent / "pos_testset_groundtruth.csv"))
    # sanity-check gold token location
    for i, row in enumerate(gold):
        locate_tokens(row["text"], row["tokens"])
        print(f"gold sentence {i}: {len(row['tokens'])} tokens OK")
