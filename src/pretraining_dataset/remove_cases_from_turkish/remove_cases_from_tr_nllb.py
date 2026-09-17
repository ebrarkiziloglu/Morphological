#!/usr/bin/env python3
"""
Pipeline
--------
0. Load the TULAP POS tagger from: Morphological/POS/tools/tulap/postagger.py
    from postagger import load_tulap_tagger
    tulap_tagger = load_tulap_tagger(device)
1. Load the Zemberek as:
     import remove_case_suffixes_from_tr_zemberek as zemb
2. Load the NLLB file (one sentence per line) from Morphological/data/NLLB_10M_subset/NLLB_10M.tr.
3. POS-tag each sentence with tulap_tagger
4. Analyse each whitespace token with Zemberek (same analyser as
   ``remove_case_suffixes_from_tr_zemberek.py``), choosing the parse whose
   POS matches the tagger (a NOUN tag can pick a noun parse that is not
   Zemberek's top-ranked analysis).
4. Drop every case marker (Dat/Loc/Abl/Acc/Gen), including those in the
   middle of the word, and the locative/genitive relative ``-ki``, from the
   morphological analysis (not from the surface string). Keep plural (A3pl)
   and possessive suffixes. Re-generate the word with Zemberek's
   ``word_generator.generate(item, morphemes)``.
5. Write a Moses-style file, default ``NLLB_small_stripped.tr``. but also take the output file path as an argument.
6. For every modified token, count case-suffix (Dat/Loc/Abl/Acc/Gen) and
   relative ``-ki`` morphemes, then print corpus-wide histograms at the end
   (also written next to the output as ``<output>.stats.txt``). Unique
   ``original → stripped`` pairs for each joint (case-suffix, -ki) group,
   with Zemberek analyses of both forms, are written under
   ``<output>.joint_groups/``.

Examples
--------
    evdekinden    -> ev
    evdekilerden  -> evler
    evlerdeki     -> evler
    kitabımı      -> kitabım
    kitabı        -> kitap

Usage (from the Morphological repo root)::

    python src/pretraining_dataset/remove_cases_from_turkish/main.py

    python src/pretraining_dataset/remove_cases_from_turkish/main.py \\
        --input data/NLLB_10M_subset/NLLB_10M.tr \\
        --output data/NLLB_10M_subset/NLLB_10M_stripped.tr
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TULAP_DIR = REPO_ROOT / "POS" / "tools" / "tulap"

DEFAULT_INPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_small.tr"
# DEFAULT_INPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_10M.tr"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_small_stripped.tr"

sys.path.insert(0, str(TULAP_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from postagger import load_tulap_tagger  # noqa: E402
import remove_case_suffixes_from_tr_zemberek as zemb  # noqa: E402
from zemberek.core.turkish import Turkish  # noqa: E402

# UD POS tag -> Zemberek primary/last-IG POS
UD_TO_ZEMBEREK = {
    "NOUN": {"Noun"},
    "PROPN": {"Noun"},
    "VERB": {"Verb"},
    "ADJ": {"Adj"},
    "ADV": {"Adv"},
    "PRON": {"Pron"},
    "ADP": {"Postp"},
    "CCONJ": {"Conj"},
    "SCONJ": {"Conj"},
    "DET": {"Det"},
    "NUM": {"Num"},
    "AUX": {"Verb"},
    "PART": {"Postp", "Conj"},
    "INTJ": {"Interj"},
    "PUNCT": {"Punc"},
}

POSSESSIVE_TAGS = {"P1sg", "P2sg", "P3sg", "P1pl", "P2pl", "P3pl"}
LEMMA_RE = re.compile(
    r"\[([^\s\]:]+):(?:Noun|Adj|Pron)",
)

# Apostrophes used in Turkish proper-noun inflection (İsrail'in, Amerika'daki)
_WORD_EXTRA = set("'’‘%")


@dataclass
class CorpusStats:
    """Token-level suffix counts for words whose surface actually changed."""

    n_sentences: int = 0
    n_empty_sentences: int = 0
    n_failed_sentences: int = 0
    n_tokens: int = 0
    n_modified: int = 0
    n_case_suffixes: int = 0
    n_ki_suffixes: int = 0
    case_hist: Counter[int] = field(default_factory=Counter)
    ki_hist: Counter[int] = field(default_factory=Counter)
    joint_hist: Counter[tuple[int, int]] = field(default_factory=Counter)
    joint_words: dict[tuple[int, int], Counter[tuple[str, str, str, str]]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def add_token(
        self,
        original: str,
        stripped: str,
        n_case: int,
        n_ki: int,
        original_analysis: str = "",
        stripped_analysis: str = "",
    ) -> None:
        self.n_tokens += 1
        if stripped == original:
            return
        self.n_modified += 1
        self.n_case_suffixes += n_case
        self.n_ki_suffixes += n_ki
        self.case_hist[n_case] += 1
        self.ki_hist[n_ki] += 1
        self.joint_hist[(n_case, n_ki)] += 1
        self.joint_words[(n_case, n_ki)][
            (original, stripped, original_analysis, stripped_analysis)
        ] += 1

    def report(self) -> str:
        def pct(part: int, whole: int) -> str:
            if whole <= 0:
                return "0.0%"
            return f"{100.0 * part / whole:.1f}%"

        lines = [
            "Corpus statistics",
            "-----------------",
            f"sentences:          {self.n_sentences:,}",
            f"  empty:            {self.n_empty_sentences:,}",
            f"  failed:           {self.n_failed_sentences:,}",
            f"tokens:             {self.n_tokens:,}",
            f"modified tokens:    {self.n_modified:,} ({pct(self.n_modified, self.n_tokens)} of tokens)",
            "",
            'case-suffix class (Dat/Loc/Abl/Acc/Gen) on modified tokens:',
            f"  total suffixes:   {self.n_case_suffixes:,}",
        ]
        if self.n_modified:
            lines.append(
                f"  mean per modified token: {self.n_case_suffixes / self.n_modified:.3f}"
            )
        if self.case_hist:
            lines.append("  tokens by case-suffix count:")
            for n in sorted(self.case_hist):
                c = self.case_hist[n]
                lines.append(f"    {n} case-suffix: {c:,}  ({pct(c, self.n_modified)})")
        lines += [
            "",
            'relative "-ki" class on modified tokens:',
            f"  total suffixes:   {self.n_ki_suffixes:,}",
        ]
        if self.n_modified:
            lines.append(
                f"  mean per modified token: {self.n_ki_suffixes / self.n_modified:.3f}"
            )
        if self.ki_hist:
            lines.append("  tokens by -ki count:")
            for n in sorted(self.ki_hist):
                c = self.ki_hist[n]
                lines.append(f"    {n} -ki: {c:,}  ({pct(c, self.n_modified)})")
        if self.joint_hist:
            lines += [
                "",
                "joint (case-suffix count, -ki count) -> modified tokens:",
            ]
            for (n_case, n_ki), c in sorted(self.joint_hist.items()):
                lines.append(
                    f"    ({n_case} case-suffix, {n_ki} -ki): {c:,}  ({pct(c, self.n_modified)})"
                )
        return "\n".join(lines) + "\n"

    def write_joint_group_files(self, output_path: Path) -> list[Path]:
        """Write one TSV per joint (case-suffix, -ki) group of modified tokens."""
        if not self.joint_words:
            return []

        group_dir = output_path.with_name(output_path.name + ".joint_groups")
        group_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for (n_case, n_ki), counts in sorted(self.joint_words.items()):
            group_path = group_dir / f"{n_case}case_{n_ki}ki.txt"
            n_tokens = self.joint_hist[(n_case, n_ki)]
            n_types = len(counts)
            ranked = sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2], item[0][3]),
            )
            lines = [
                f"# ({n_case} case-suffix, {n_ki} -ki): "
                f"{n_tokens:,} tokens, {n_types:,} unique original → stripped pairs",
                "count\toriginal\tstripped\toriginal_analysis\tstripped_analysis",
            ]
            lines.extend(
                "\t".join(
                    (
                        str(count),
                        _tsv_cell(original),
                        _tsv_cell(stripped),
                        _tsv_cell(original_analysis),
                        _tsv_cell(stripped_analysis),
                    )
                )
                for (
                    original,
                    stripped,
                    original_analysis,
                    stripped_analysis,
                ), count in ranked
            )
            group_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(group_path)
        return written


def _tsv_cell(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ").replace("\r", "")


def count_case_and_ki(parse) -> tuple[int, int]:
    tags = zemb.morpheme_ids(parse)
    n_case = sum(1 for tag in tags if tag in zemb.TARGET_CASES)
    n_ki = sum(1 for tag in tags if tag == "Rel")
    return n_case, n_ki


def peel_punct(token: str) -> tuple[str, str, str]:
    """Split leading/trailing punctuation, keeping apostrophes on the core."""

    def is_core_char(char: str) -> bool:
        return char.isalnum() or char in _WORD_EXTRA

    start, end = 0, len(token)
    while start < end and not is_core_char(token[start]):
        start += 1
    while end > start and not is_core_char(token[end - 1]):
        end -= 1
    return token[:start], token[start:end], token[end:]


def restore_word_case(original: str, stripped: str) -> str:
    """Copy casing from ``original`` using Turkish i/İ maps."""
    if not stripped:
        return stripped
    if original.isupper():
        return stripped.translate(Turkish.alphabet.upper_map).upper()
    if original[:1].isupper():
        return Turkish.capitalize(stripped)
    return stripped


def zemberek_pos_labels(parse) -> set[str]:
    labels = {parse.item.primary_pos.short_form}
    last = zemb.last_pos_tag(zemb.morpheme_ids(parse))
    if last:
        labels.add(last)
    return labels


def parse_matches_pos(parse, ud_pos: str | None) -> bool:
    if not ud_pos or ud_pos == "X":
        return False
    targets = UD_TO_ZEMBEREK.get(ud_pos, set())
    if not targets:
        return False
    labels = zemberek_pos_labels(parse)
    if ud_pos == "PROPN":
        return "Noun" in labels and parse.item.secondary_pos.short_form == "Prop"
    if ud_pos == "NOUN":
        return "Noun" in labels and parse.item.secondary_pos.short_form != "Prop"
    return bool(labels & targets)


def choose_parse(parses: list, ud_pos: str | None):
    """Pick the Zemberek parse that agrees with the POS tag.

    Among matches, prefer analyses that contain a case / Rel (so *oyun* is
    treated as *oy* + Gen/P2sg "your vote", not the lemma *oyun* "game").
    Then prefer a longer case surface (*masa+da* over *masat+a*), then a
    longer lemma (*deniz+Acc* over *de+P2pl+Acc*), then Zemberek's own rank.
    """

    def score(item) -> tuple:
        rank, parse = item
        tags = zemb.morpheme_ids(parse)
        spos = parse.item.secondary_pos.short_form
        primary = parse.item.primary_pos.short_form
        lemma_len = len(parse.item.lemma or "")
        stem_len = 0
        last_case_len = 0
        if parse.morpheme_data_list:
            stem_len = len(parse.morpheme_data_list[0].surface or "")
            last = parse.morpheme_data_list[-1]
            if last.morpheme.id_ in zemb.TARGET_CASES:
                last_case_len = len(last.surface or "")
        has_case = any(tag in zemb.TARGET_CASES or tag == "Rel" for tag in tags)
        targets = UD_TO_ZEMBEREK.get(ud_pos or "", set())
        primary_fit = int(primary in targets) if targets else 0
        if ud_pos == "PROPN":
            prop_fit = int(spos == "Prop")
        elif ud_pos == "NOUN":
            prop_fit = int(spos != "Prop")
        else:
            prop_fit = 1
        return (
            int(parse_matches_pos(parse, ud_pos)),
            primary_fit,
            int(has_case),
            prop_fit,
            last_case_len,
            lemma_len,
            -rank,
            int("P3sg" in tags),
            stem_len,
        )

    return max(enumerate(parses), key=score)[1]


def lemma_from_parse(parse) -> str:
    lemma = parse.item.lemma or ""
    match = LEMMA_RE.search(parse.format_string())
    if match:
        lemma = match.group(1)
    return zemb.unvoice_final_consonant(lemma.lower())


def morphemes_without_case_and_rel(parse) -> tuple:
    """Morphemes to keep after dropping case and Rel ``-ki``.

    Rel is derivational (``Loc/Gen + ki → Adj``), so the Adj | Zero → Noun
    chain after it cannot be generated without Rel. Keep pre-Rel stem,
    derivation, number, and possessive; if plural (A3pl) appears after
    ``-ki`` (evdekiler), promote it onto the original stem.
    """
    kept: list = []
    seen_rel = False
    post_rel_a3pl = None
    for md in parse.morpheme_data_list:
        tag = md.morpheme.id_
        if tag == "Rel":
            seen_rel = True
            continue
        if tag in zemb.TARGET_CASES:
            continue
        if seen_rel:
            if tag == "A3pl":
                post_rel_a3pl = md.morpheme
            continue
        kept.append(md.morpheme)
    if post_rel_a3pl is not None:
        kept = [m for m in kept if m.id_ != "A3sg"]
        if all(m.id_ != "A3pl" for m in kept):
            kept.append(post_rel_a3pl)
    return tuple(kept)


def format_kept_analysis(parse, kept_morphemes) -> str:
    """Compact analysis string for leftover morphemes when generation has no Result."""
    lemma = parse.item.lemma or ""
    pos = parse.item.primary_pos.short_form
    ids = "+".join(m.id_ for m in kept_morphemes)
    if ids:
        return f"[{lemma}:{pos}] {ids}"
    return f"[{lemma}:{pos}]"


def _pick_generated_result(results, kept_morphemes):
    """Prefer an exact morpheme match, then fewer extra morphemes, then shorter."""
    kept_ids = tuple(m.id_ for m in kept_morphemes)

    def sort_key(result) -> tuple:
        ids = tuple(m.id_ for m in result.analysis.get_morphemes())
        mismatch = sum(a != b for a, b in zip(ids, kept_ids))
        extra = abs(len(ids) - len(kept_ids))
        exact = int(ids != kept_ids)
        return (exact, extra + mismatch, len(result.surface))

    return min(results, key=sort_key)


def generate_from_morphemes(parse, kept_morphemes):
    """Re-generate a surface from ``parse.item`` + remaining morphemes.

    Same idea as Java Zemberek::

        morphology.getWordGenerator().generate(item, "A3sg", "P1sg", "Dat")

    Returns ``(surface, analysis_string)`` or ``None``.
    """
    if not kept_morphemes:
        return None
    item = parse.item.reference_item or parse.item
    try:
        results = zemb.morphology.word_generator.generate(
            item=item, morphemes=kept_morphemes
        )
    except Exception:  # noqa: BLE001 — generator can fail on odd morpheme sequences
        return None
    if not results:
        return None
    result = _pick_generated_result(results, kept_morphemes)
    return result.surface, result.analysis.format_string()


def _strip_from_surfaces(word: str, parse) -> str:
    """Fallback if generation fails: concat leftover surfaces / dictionary lemma."""
    seen_rel = False
    kept_surfaces: list[str] = []
    kept_non_stem = False
    for md in parse.morpheme_data_list:
        tag = md.morpheme.id_
        surface = md.surface or ""
        if tag == "Rel":
            seen_rel = True
            continue
        if tag in zemb.TARGET_CASES:
            continue
        if seen_rel and tag in POSSESSIVE_TAGS:
            continue
        if not surface:
            continue
        kept_surfaces.append(surface)
        if tag not in zemb.POS_TAGS:
            kept_non_stem = True
    if not kept_surfaces:
        return word
    if not kept_non_stem:
        lemma = lemma_from_parse(parse)
        if not lemma:
            return word
        return restore_word_case(word, lemma)
    return restore_word_case(word, "".join(kept_surfaces))


def strip_cases_keep_plural_possessive(word: str, parse) -> tuple[str, str]:
    """Drop case and Rel ``-ki`` from the analysis, then re-generate the word.

    Returns ``(stripped_surface, stripped_analysis)``.
    """
    morphs = parse.morpheme_data_list
    if not morphs:
        return word, ""

    tags = zemb.morpheme_ids(parse)
    if not any(tag in zemb.TARGET_CASES or tag == "Rel" for tag in tags):
        return word, parse.format_string()

    kept = morphemes_without_case_and_rel(parse)
    generated = generate_from_morphemes(parse, kept)
    if generated:
        surface, analysis = generated
        return restore_word_case(word, surface), analysis
    return _strip_from_surfaces(word, parse), format_kept_analysis(parse, kept)


def strip_token(token: str, ud_pos: str | None) -> tuple[str, int, int, str, str]:
    prefix, core, suffix = peel_punct(token)
    if not core:
        return token, 0, 0, "", ""

    analysis = zemb.morphology.analyze(core)
    parses = [p for p in analysis if not p.is_unknown()]
    if not parses:
        return token, 0, 0, "", ""

    parse = choose_parse(parses, ud_pos)
    n_case, n_ki = count_case_and_ki(parse)
    stripped, stripped_analysis = strip_cases_keep_plural_possessive(core, parse)
    original_analysis = parse.format_string()
    if not stripped:
        return token, n_case, n_ki, original_analysis, stripped_analysis
    return prefix + stripped + suffix, n_case, n_ki, original_analysis, stripped_analysis


def pos_tag_words(tagger, words: list[str]) -> list[str]:
    """Map TULAP predictions onto original whitespace tokens via word_ids.

    ``PosTagger.pos_tag`` returns WordPiece first-pieces and can split off
    punctuation, so it is not a 1-1 map onto ``sentence.split()``. Zemberek
    analyses the original whitespace tokens, so each of those needs one tag.
    """
    import torch

    if not words:
        return []

    encoded = tagger.tokenizer(
        words,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )
    word_ids = tagger.tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    ).word_ids()

    device = next(tagger.model.parameters()).device
    encoded = encoded.to(device)

    with torch.no_grad():
        logits = tagger.model(**encoded).logits
    pred_ids = torch.argmax(logits, dim=2)[0].tolist()

    tags = [None] * len(words)
    for idx, word_id in enumerate(word_ids):
        if word_id is None or tags[word_id] is not None:
            continue
        if idx >= len(pred_ids):
            break
        tags[word_id] = tagger.id2label.get(pred_ids[idx], "X")
    return [tag or "X" for tag in tags]


def strip_sentence(tagger, sentence: str, stats: CorpusStats) -> str:
    words = sentence.split()
    if not words:
        return sentence
    tags = pos_tag_words(tagger, words)
    stripped_words: list[str] = []
    for word, tag in zip(words, tags):
        stripped, n_case, n_ki, original_analysis, stripped_analysis = strip_token(
            word, tag
        )
        stats.add_token(
            word, stripped, n_case, n_ki, original_analysis, stripped_analysis
        )
        stripped_words.append(stripped)
    return " ".join(stripped_words)


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def process_file(
    input_path: Path,
    output_path: Path,
    device: str,
    max_lines: int | None = None,
) -> None:
    from tqdm import tqdm

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading TULAP POS tagger...", flush=True)
    tulap_tagger = load_tulap_tagger(device)
    print("Zemberek analyser ready.", flush=True)

    n_lines = count_lines(input_path)
    if max_lines is not None:
        n_lines = min(n_lines, max_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = CorpusStats()

    with (
        input_path.open("r", encoding="utf-8") as in_f,
        output_path.open("w", encoding="utf-8") as out_f,
    ):
        for i, raw_line in enumerate(
            tqdm(in_f, total=n_lines, desc="Stripping cases", unit="sent")
        ):
            if max_lines is not None and i >= max_lines:
                break
            stats.n_sentences += 1
            text = raw_line.rstrip("\n")
            if not text.strip():
                out_f.write("\n")
                stats.n_empty_sentences += 1
                continue
            try:
                out_f.write(strip_sentence(tulap_tagger, text, stats) + "\n")
            except Exception as exc:  # noqa: BLE001 — keep the run going
                out_f.write(text + "\n")
                stats.n_failed_sentences += 1
                print(
                    f"Line {i + 1} failed ({type(exc).__name__}: {exc}); wrote original",
                    flush=True,
                )

    elapsed = time.time() - t0
    n_done = stats.n_sentences
    rate = n_done / elapsed if elapsed > 0 else 0.0
    print(
        f"Done in {elapsed:.1f}s ({rate:.1f} sent/s). "
        f"ok={n_done - stats.n_empty_sentences - stats.n_failed_sentences:,} "
        f"empty={stats.n_empty_sentences:,} failed={stats.n_failed_sentences:,}",
        flush=True,
    )
    print(f"Wrote {output_path}", flush=True)
    report = stats.report()
    print(report, end="", flush=True)
    stats_path = output_path.with_name(output_path.name + ".stats.txt")
    stats_path.write_text(report, encoding="utf-8")
    print(f"Wrote {stats_path}", flush=True)
    group_paths = stats.write_joint_group_files(output_path)
    for group_path in group_paths:
        print(f"Wrote {group_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip Turkish case suffixes from NLLB using TULAP POS + Zemberek",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if _cuda_available() else "cpu",
    )
    parser.add_argument("--max-lines", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_file(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        max_lines=args.max_lines,
    )


if __name__ == "__main__":
    main()
