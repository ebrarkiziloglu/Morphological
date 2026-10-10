#!/usr/bin/env python3
"""
Pipeline
--------
0. Load the TULAP POS tagger from: Morphological/Turkish_POS/tools/tulap/postagger.py
    from postagger import load_tulap_tagger
    tulap_tagger = load_tulap_tagger(device)
1. Load the Zemberek as:
     import remove_case_suffixes_from_tr_zemberek as zemb
2. Load the NLLB file (one sentence per line) from Morphological/data/NLLB_10M_subset/NLLB_10M.tr.
3. POS-tag each sentence with tulap_tagger
4. Analyse each whitespace token with Zemberek (same analyser as
   ``remove_case_suffixes_from_tr_zemberek.py``) and disambiguate the
   sentence in context. Use Zemberek's best parse if its POS matches the
   tagger, otherwise a parse whose POS matches (a NOUN tag can pick a noun
   parse that is not Zemberek's best analysis).
4. Drop morphemes from the morphological analysis (not from the surface
   string) according to ``--mode``, then re-generate the word with Zemberek's
   ``word_generator.generate(item, morphemes)``:
   - ``case`` (default): every case marker (Dat/Loc/Abl/Acc/Gen and
     instrumental ``-(y)lA``, equative ``-cA``), including those in the
     middle of the word, and the locative/genitive relative ``-ki``. Keep
     plural (A3pl) and possessive suffixes.
   - ``postp``: ``case`` + delete postposition words (için, ile, gibi, göre,
     kadar, sonra, ...: tokens whose chosen parse is Zemberek ``Postp``).
     Their punctuation stays on the neighbouring words, so the output has
     fewer tokens per line than the input.
   - ``nominal``: all nominal morphology. Nominal roots lose inflection
     (case, plural, possessive, -ki), noun/adj derivation (-lIk, -CI, -lI,
     -sIz, ...) and nominal predicates (copula, -DI, -mIş, person:
     evdeyiz -> ev). Nominalised verbs (-mAk, -mA, -DIk, -(y)An, ...) are cut
     back to the verb stem, keeping voice and negation. Noun -> verb
     derivation (-lAş, -lAn), finite verbs and converbs are kept.
   - ``nominal_infl``: all nominal inflection, but every derivation is kept.
     Every nominal inflectional group (also one derived from a verb) loses
     case, plural, possessive, -ki and nominal predicates (copula, -DI, -mIş,
     person): evsizlerin -> evsiz, gelmesini -> gelme, evdeyiz -> ev.
     Verbal inflection (finite verbs, converbs) is kept.
5. Write a Moses-style file, default
   ``NLLB_10M_stripped{,_postp,_nominal,_nominal_infl}.tr``
   by mode, but also take the output file path as an argument.
6. For every modified token, count case-suffix (Dat/Loc/Abl/Acc/Gen/Ins/Equ)
   and relative ``-ki`` morphemes, then print corpus-wide histograms at the end
   (also written next to the output as ``<output>.stats.txt``, together with
   removed morphemes and deleted postpositions). Unique
   ``original → stripped`` pairs for each joint (case-suffix, -ki) group,
   with Zemberek analyses of both forms, are written under
   ``<output>.joint_groups/``.

Examples (case / nominal / nominal_infl)
----------------------------------------
    evdekinden       -> ev           / ev       / ev
    evlerdeki        -> evler        / ev       / ev
    kitabımı         -> kitabım      / kitap    / kitap
    arkadaşlarımla   -> arkadaşlarım / arkadaş  / arkadaş
    evsizlerin       -> evsizler     / ev       / evsiz
    evdeyiz          -> eviz         / ev       / ev
    gelmesini        -> gelmesi      / gel      / gelme
    yapılmasını      -> yapılması    / yapıl    / yapılma
    güzelleşmesini   -> güzelleşmesi / güzelleş / güzelleşme
    postp: "Ev için, ona göre geldi." -> "Ev, ona geldi."

Usage (from the Morphological repo root)::

    python src/pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py

    python src/pretraining_dataset/remove_cases_from_turkish/remove_cases_from_tr_nllb.py \\
        --mode nominal \\
        --input data/NLLB_10M_subset/NLLB_10M.tr \\
        --output data/NLLB_10M_subset/NLLB_10M_stripped_nominal.tr
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
TULAP_DIR = REPO_ROOT / "Turkish_POS" / "tools" / "tulap"

# DEFAULT_INPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_small.tr"
DEFAULT_INPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_10M.tr"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "NLLB_10M_subset" / "NLLB_10M_stripped.tr"
DEFAULT_OUTPUTS = {
    "case": DEFAULT_OUTPUT,
    "postp": DEFAULT_OUTPUT.with_name("NLLB_10M_stripped_postp.tr"),
    "nominal": DEFAULT_OUTPUT.with_name("NLLB_10M_stripped_nominal.tr"),
    "nominal_infl": DEFAULT_OUTPUT.with_name("NLLB_10M_stripped_nominal_infl.tr"),
}

sys.path.insert(0, str(TULAP_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from postagger import load_tulap_tagger  # noqa: E402
import remove_case_suffixes_from_tr_zemberek as zemb  # noqa: E402
from zemberek.core.turkish import Turkish  # noqa: E402
from zemberek.morphology.morphotactics.turkish_morphotactics import (  # noqa: E402
    TurkishMorphotactics,
)

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
# Every Zemberek case: Dat/Loc/Abl/Acc/Gen + instrumental -(y)lA + equative -cA
ALL_CASE_TAGS = zemb.TARGET_CASES | {"Ins", "Equ"}

# "postp" mode: deleted whatever their parse (ile is often tagged Conj)
ALWAYS_POSTPOSITIONS = {"ile"}

# "nominal" mode
NOMINAL_ROOT_POS = {"Noun", "Adj", "Pron", "Num"}
# Noun/Adj -> Verb derivations kept as part of the stem (güzelleş-, kitaplaş-)
NOUN_TO_VERB_DERIVS = {"Become", "Acquire"}
# Polarity stays on a verb stem cut back from a nominalisation (gelmediği -> gelme)
VERB_STEM_INFLECTIONS = {"Neg", "Unable"}

# --mode values:
#   case:    every case (incl. Ins, Equ) + relative -ki; keep plural, possessive
#   postp:   case + delete postposition words (için, ile, gibi, göre, ...)
#   nominal: all nominal morphology (inflection, nominal predicates, noun/adj
#            derivation); nominalised verbs back to the verb stem
#   nominal_infl: all nominal inflection (incl. nominal predicates); keep
#            every derivation and all verbal inflection
STRIP_MODES = ("case", "postp", "nominal", "nominal_infl")
# "nominal_infl" mode: IG POS whose inflection is removed, and that inflection
NOMINAL_IG_POS = {"Noun", "Adj", "Pron", "Num"}
NOMINAL_INFLECTIONS = {"A3pl"} | POSSESSIVE_TAGS | ALL_CASE_TAGS
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
    selection: Counter[str] = field(default_factory=Counter)
    dropped_tags: Counter[str] = field(default_factory=Counter)
    deleted_postp: Counter[str] = field(default_factory=Counter)
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
            "parse selection (analysed tokens):",
            *(
                f"  {method}: {c:,}  ({pct(c, sum(self.selection.values()))})"
                for method, c in self.selection.most_common()
            ),
            "",
            f"deleted postpositions: {sum(self.deleted_postp.values()):,}",
            *(f"  {lemma}: {c:,}" for lemma, c in self.deleted_postp.most_common()),
            "",
            "removed morphemes on modified tokens (non-empty surface):",
            *(
                f"  {tag}: {c:,}"
                for tag, c in self.dropped_tags.most_common()
            ),
            "",
            'case-suffix class (Dat/Loc/Abl/Acc/Gen/Ins/Equ) on modified tokens:',
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
    n_case = sum(1 for tag in tags if tag in ALL_CASE_TAGS)
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


def choose_parse(parses: list, best, ud_pos: str | None) -> tuple:
    """Pick the Zemberek parse, using the contextual best analysis and the POS tag.

    1. Zemberek's sentence-level disambiguated analysis (``best``), if it
       agrees with the POS tag ("disambiguated").
    2. Otherwise a parse that agrees with the POS tag ("pos_override").
    3. Otherwise ``best`` ("no_pos_match"), or the top fallback-ranked parse if
       disambiguation failed ("no_disambiguation").

    The fallback ranking never prefers a parse *because* it has a case: that
    over-segmented real words (para -> par+Dat, Türkiye -> Türki+Dat,
    şeyi -> şe+Acc). It prefers non-abbreviation parses, then a longer lemma
    (fewer segments), then Zemberek's order.

    Returns ``(parse, selection_method)``.
    """
    if best is not None:
        best_str = best.format_string()
        best = next((p for p in parses if p.format_string() == best_str), None)

    def fallback_score(item) -> tuple:
        rank, parse = item
        return (
            int(parse.item.secondary_pos.short_form != "Abbrv"),
            len(parse.item.lemma or ""),
            -rank,
        )

    def top(candidates: list):
        return max(enumerate(candidates), key=fallback_score)[1]

    if best is not None and parse_matches_pos(best, ud_pos):
        chosen, method = best, "disambiguated"
    elif matching := [p for p in parses if parse_matches_pos(p, ud_pos)]:
        chosen, method = top(matching), "pos_override"
    elif best is not None:
        chosen, method = best, "no_pos_match"
    else:
        chosen, method = top(parses), "no_disambiguation"
    return prefer_p3sg_over_p2sg(chosen, parses), method


def prefer_p3sg_over_p2sg(parse, parses: list):
    """Swap a P2sg+case parse for its P3sg twin (olduğunu: ol+duğ+u+nu, not +un+u).

    The surfaces are identical and Zemberek's disambiguator systematically
    picks P2sg ("your"), which is rare in NLLB's mostly third-person text.
    """
    tags = zemb.morpheme_ids(parse)
    if "P2sg" not in tags:
        return parse
    target = ["P3sg" if tag == "P2sg" else tag for tag in tags]
    for other in parses:
        if other.item.lemma == parse.item.lemma and zemb.morpheme_ids(other) == target:
            return other
    return parse


def lemma_from_parse(parse) -> str:
    lemma = parse.item.lemma or ""
    match = LEMMA_RE.search(parse.format_string())
    if match:
        lemma = match.group(1)
    return zemb.unvoice_final_consonant(lemma.lower())


def case_dropped_flags(parse) -> list[bool]:
    """For each morpheme of ``parse``: is it removed in "case"/"postp" mode?

    Every case (Dat/Loc/Abl/Acc/Gen/Ins/Equ), including those in the middle of
    the word, Rel ``-ki`` and everything after Rel (its Adj | Zero → … chain
    cannot be generated without Rel).
    """
    flags: list[bool] = []
    seen_rel = False
    for md in parse.morpheme_data_list:
        tag = md.morpheme.id_
        seen_rel |= tag == "Rel"
        flags.append(seen_rel or tag in ALL_CASE_TAGS)
    return flags


def morphemes_to_keep(parse) -> tuple:
    """Morphemes to re-generate from after dropping case and Rel ``-ki``.

    If plural (A3pl) appears after ``-ki`` (evdekiler), promote it onto the
    original stem (-> evler).
    """
    kept: list = []
    seen_rel = False
    post_rel_a3pl = None
    for md, drop in zip(parse.morpheme_data_list, case_dropped_flags(parse)):
        tag = md.morpheme.id_
        seen_rel |= tag == "Rel"
        if not drop:
            kept.append(md.morpheme)
        elif tag == "A3pl" and seen_rel:
            post_rel_a3pl = md.morpheme
    if post_rel_a3pl is not None:
        kept = [m for m in kept if m.id_ != "A3sg"]
        if all(m.id_ != "A3pl" for m in kept):
            kept.append(post_rel_a3pl)
    return tuple(kept)


@dataclass
class NominalPlan:
    """What "nominal" mode does with one parse."""

    flags: list[bool]  # per morpheme: removed?
    kept: tuple  # morphemes of the remaining stem (for generation)
    verbal: bool  # the remaining stem is a verb stem (else a nominal root)


def nominal_plan(parse) -> NominalPlan | None:
    """Plan for removing all nominal morphology, or None to leave the word as is.

    Walks the inflectional groups (IGs) of the parse:

    - Nominal root (Noun/Adj/Pron/Num): drop its inflection (case, plural,
      possessive, -ki) and noun/adj -> noun/adj/adv derivations (-lIk, -CI,
      -lI, -sIz, -cIk, -sAl, -ImsI, -CA ...), so the result is the bare root:
      evsizlerin -> ev, güzelliğini -> güzel, onları -> o.
    - Nominal predicate (Zero -> Verb after a nominal IG: copula, -DI, -mIş,
      -sA, -ken, person) is cut: evdeyiz -> ev, parçasıdır -> parça.
    - Noun -> verb derivation (-lAş, -lAn) is kept; the word continues as a verb.
    - Verb IG followed by a nominaliser (-mAk, -mA, -(y)Iş, -DIk, -(y)AcAk,
      -(y)An, -mIş, ...) is cut back to the verb stem, keeping verb -> verb
      derivations (passive, causative, ...) and negation:
      gelmesini -> gel, yapılmasını -> yapıl, gelmediğini -> gelme.

    Finite verbs, converbs (gelince, gelip) and non-nominal roots (Adv, Conj,
    Postp, Det, ...) are left unchanged (None).
    """
    mds = parse.morpheme_data_list
    if not mds:
        return None
    root = mds[0].morpheme.id_
    if root not in NOMINAL_ROOT_POS and root != "Verb":
        return None

    flags = [True] * len(mds)
    flags[0] = False
    kept = [mds[0].morpheme]
    verbal = root == "Verb"
    cut = False
    i = 1
    while i < len(mds):
        morpheme = mds[i].morpheme
        if not morpheme.derivational_:
            # Inflection: verbal polarity stays on the stem, anything else is
            # removed (nominal) or ends a finite verb (handled below via `cut`).
            if verbal and morpheme.id_ in VERB_STEM_INFLECTIONS:
                flags[i] = False
                kept.append(morpheme)
            i += 1
            continue

        target = mds[i + 1].morpheme.id_ if i + 1 < len(mds) else None
        if verbal and target == "Verb":
            # verb -> verb derivation (Pass, Caus, Able, ...): part of the stem
            keep = True
        elif verbal and target in ("Noun", "Adj"):
            cut = True  # nominalisation: cut back to the verb stem
            break
        elif verbal:
            return None  # converb (gelince, gelip): not nominal morphology
        elif morpheme.id_ in NOUN_TO_VERB_DERIVS:
            # The generator needs number agreement on the nominal IG.
            if root in ("Noun", "Pron"):
                kept.append(TurkishMorphotactics.a3sg)
            keep = verbal = True
        elif target == "Verb":
            cut = True  # nominal predicate (Zero -> Verb)
            break
        else:
            keep = False  # noun/adj -> noun/adj/adv derivation: drop
        if keep and target is not None:
            flags[i] = flags[i + 1] = False
            kept += [morpheme, mds[i + 1].morpheme]
        i += 2 if target is not None else 1

    if verbal and not cut:
        return None  # finite verb: no nominal morphology to remove
    if not any(flag and md.surface for flag, md in zip(flags, mds)):
        return None
    return NominalPlan(flags=flags, kept=tuple(kept), verbal=verbal)


def generate_verb_stem(parse, kept_morphemes) -> str | None:
    """Verb stem surface = 2sg imperative of the kept verb morphemes (gel, yapıl, gelme)."""
    generated = generate_from_morphemes(
        parse, (*kept_morphemes, TurkishMorphotactics.imp, TurkishMorphotactics.a2sg)
    )
    return generated[0] if generated else None


def generate_nominal_root(parse) -> str:
    """Bare dictionary surface of a nominal root (kitap, mekan, o, ben)."""
    root = parse.morpheme_data_list[0].morpheme
    morphemes = (root, TurkishMorphotactics.a3sg) if root.id_ in ("Noun", "Pron") else (root,)
    generated = generate_from_morphemes(parse, morphemes)
    if generated:
        return generated[0]
    return parse.item.lemma or ""


def strip_nominal(word: str, parse) -> tuple[str, str, tuple[str, ...]]:
    """ "nominal" mode: ``(stripped_surface, stripped_analysis, dropped_ids)``."""
    plan = nominal_plan(parse)
    if plan is None:
        return word, parse.format_string(), ()
    dropped = tuple(
        md.morpheme.id_
        for md, flag in zip(parse.morpheme_data_list, plan.flags)
        if flag and md.surface
    )
    analysis = format_kept_analysis(parse, plan.kept)
    if plan.verbal:
        surface = generate_verb_stem(parse, plan.kept) or parse.item.root
        return restore_word_case(word, surface), analysis, dropped
    rebuilt = _strip_after_apostrophe(word, parse, plan.flags)
    if rebuilt is not None:
        return rebuilt, analysis, dropped
    return restore_word_case(word, generate_nominal_root(parse)), analysis, dropped


def nominal_infl_plan(parse) -> tuple[list[bool], tuple] | None:
    """``(flags, kept)`` for removing nominal inflection only, or None to leave the word.

    Walks the inflectional groups (IGs) of the parse and keeps every
    derivation (-lIk, -CI, -sIz, -mA, -DIk, -(y)An, -lAş, Pass, Caus, ...):

    - Nominal IG (Noun/Adj/Pron/Num, also one derived from a verb): plural,
      possessive and case are removed. A3pl becomes A3sg, which the generator
      needs as number agreement: kitapçılarımızdan -> kitapçı,
      gelmesini -> gelme, geldiğini -> geldik, onları -> o.
    - Rel ``-ki`` or a nominal predicate (Zero -> Verb after a nominal IG:
      copula, -DI, -mIş, -sA, -ken, person) cuts the rest of the word:
      evdekiler -> ev, evdeyiz -> ev.
    - Verb IG: inflection kept (finite verbs, converbs, negation unchanged).
    """
    mds = parse.morpheme_data_list
    if not mds:
        return None
    flags = [False] * len(mds)
    kept = [mds[0].morpheme]
    ig_pos = mds[0].morpheme.id_
    i = 1
    while i < len(mds):
        morpheme = mds[i].morpheme
        tag = morpheme.id_
        nominal = ig_pos in NOMINAL_IG_POS
        if not morpheme.derivational_:
            if nominal and tag in NOMINAL_INFLECTIONS:
                flags[i] = True
                if tag == "A3pl":
                    kept.append(TurkishMorphotactics.a3sg)
            else:
                kept.append(morpheme)
            i += 1
            continue

        target = mds[i + 1].morpheme.id_ if i + 1 < len(mds) else None
        if nominal and (
            tag == "Rel" or (target == "Verb" and tag not in NOUN_TO_VERB_DERIVS)
        ):
            flags[i:] = [True] * (len(mds) - i)
            break
        kept.append(morpheme)
        if target is not None:
            kept.append(mds[i + 1].morpheme)
            ig_pos = target
        i += 2 if target is not None else 1

    if not any(flag and md.surface for flag, md in zip(flags, mds)):
        return None
    return flags, tuple(kept)


def strip_nominal_infl(word: str, parse) -> tuple[str, str, tuple[str, ...]]:
    """ "nominal_infl" mode: ``(stripped_surface, stripped_analysis, dropped_ids)``."""
    plan = nominal_infl_plan(parse)
    if plan is None:
        return word, parse.format_string(), ()
    flags, kept = plan
    mds = parse.morpheme_data_list
    dropped = tuple(
        md.morpheme.id_ for md, flag in zip(mds, flags) if flag and md.surface
    )
    rebuilt = _strip_after_apostrophe(word, parse, flags)
    if rebuilt is not None:
        return rebuilt, format_kept_analysis(parse, kept), dropped
    generated = generate_from_morphemes(parse, kept)
    if generated:
        surface, analysis = generated
        return restore_word_case(word, surface), analysis, dropped
    # Generation failed: bare root, or the kept surfaces concatenated.
    if all(m.id_ in zemb.POS_TAGS or m.id_ == "A3sg" for m in kept):
        surface = generate_nominal_root(parse)
    else:
        surface = "".join(md.surface or "" for md, flag in zip(mds, flags) if not flag)
    return restore_word_case(word, surface or word), format_kept_analysis(parse, kept), dropped


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
    for md, drop in zip(parse.morpheme_data_list, case_dropped_flags(parse)):
        tag = md.morpheme.id_
        surface = md.surface or ""
        seen_rel |= tag == "Rel"
        # keep a post -ki plural surface (evdekiler -> evler)
        if drop and not (seen_rel and tag == "A3pl"):
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


def _harmonized_plural(text: str) -> str:
    """``lar``/``ler`` by the last vowel of ``text`` (Ankara'dakiler -> Ankara'lar)."""
    for char in reversed(text.lower()):
        if char in "aıou":
            return "lar"
        if char in "eiöü":
            return "ler"
    return "ler"


def _strip_after_apostrophe(
    word: str, parse, flags: list[bool], promote_rel_plural: bool = False
) -> str | None:
    """Rebuild an apostrophe word as ``<text before '>'<kept suffixes>``.

    Proper nouns, abbreviations and numbers (Pula'ya, ABD'deki, %15'inin) keep
    their stem verbatim before the apostrophe, so the surface is rebuilt from
    the original suffix surfaces instead of the generator, which fails for
    out-of-dictionary names and drops the apostrophe on numbers. ``flags``
    marks the morphemes to drop; with ``promote_rel_plural`` a plural after
    ``-ki`` moves onto the stem (Ankara'dakiler -> Ankara'lar).
    """
    for i, char in enumerate(word):
        if char in "'’‘":
            stem, apostrophe = word[:i], char
            break
    else:
        return None
    if not stem:
        return None

    suffixes: list[str] = []
    seen_rel = False
    pre_rel_plural = post_rel_plural = False
    consumed = 0  # characters of the stem covered so far
    for md, drop in zip(parse.morpheme_data_list, flags):
        if consumed < len(stem):
            # Morphemes inside the pre-apostrophe text (Deklarasyon+u’na: the
            # P3sg "u" is part of the stem text, not a suffix to re-append).
            # If such a morpheme must be dropped (nominal: Cumhuriyet+i'nin),
            # the text cannot be copied verbatim; let the caller fall back.
            if drop and md.surface:
                return None
            consumed += len(md.surface or "")
            continue
        tag = md.morpheme.id_
        seen_rel |= tag == "Rel"
        if drop:
            post_rel_plural |= seen_rel and tag == "A3pl"
            continue
        pre_rel_plural |= tag == "A3pl" and bool(md.surface)
        suffixes.append(md.surface or "")
    # The promoted plural's surface must re-harmonise with the stem.
    if promote_rel_plural and post_rel_plural and not pre_rel_plural:
        suffixes.insert(0, _harmonized_plural(stem))
    suffix = "".join(suffixes)
    return stem + apostrophe + suffix if suffix else stem


def strip_cases(word: str, parse) -> tuple[str, str, tuple[str, ...]]:
    """"case"/"postp" mode: drop every case and Rel ``-ki`` from the analysis,
    then re-generate the word.

    Returns ``(stripped_surface, stripped_analysis, dropped_ids)``.
    """
    morphs = parse.morpheme_data_list
    if not morphs:
        return word, "", ()

    flags = case_dropped_flags(parse)
    if not any(flags):
        return word, parse.format_string(), ()
    dropped = tuple(
        md.morpheme.id_
        for md, drop in zip(morphs, flags)
        if drop and (md.surface or md.morpheme.id_ == "Rel")
    )

    kept = morphemes_to_keep(parse)
    rebuilt = _strip_after_apostrophe(word, parse, flags, promote_rel_plural=True)
    if rebuilt is not None:
        return rebuilt, format_kept_analysis(parse, kept), dropped
    generated = generate_from_morphemes(parse, kept)
    if generated:
        surface, analysis = generated
        return restore_word_case(word, surface), analysis, dropped
    return _strip_from_surfaces(word, parse), format_kept_analysis(parse, kept), dropped


def is_postposition(parse) -> bool:
    """Postposition word (için, ile, gibi, göre, kadar, sonra, ...) in this context.

    ``ile`` counts in both of its readings: TULAP tags it CCONJ almost always
    (also in "garantisi ile" "with the guarantee"), and the conjunction "and"
    is the same comitative ``ile`` whose clitic form -(y)lA is stripped as Ins.
    """
    return (
        parse.item.primary_pos.short_form == "Postp"
        or parse.item.lemma in ALWAYS_POSTPOSITIONS
    )


@dataclass
class TokenResult:
    stripped: str  # "" when the whole token is deleted (postp mode)
    n_case: int = 0
    n_ki: int = 0
    original_analysis: str = ""
    stripped_analysis: str = ""
    method: str | None = None  # parse-selection method, None if not analysed
    dropped: tuple[str, ...] = ()  # morpheme ids removed by the mode
    deleted_postp: str | None = None  # lemma of a deleted postposition
    prefix: str = ""  # punctuation around the core, kept when the word is deleted
    suffix: str = ""


def strip_token(
    token: str, ud_pos: str | None, analysis=None, best=None, mode: str = "case"
) -> TokenResult:
    """Strip one whitespace token.

    ``analysis`` is Zemberek's WordAnalysis of the token core and ``best`` the
    sentence-level disambiguated parse (analysis is computed here if not given).
    """
    prefix, core, suffix = peel_punct(token)
    if not core:
        return TokenResult(token)

    if analysis is None:
        analysis = zemb.morphology.analyze(core)
    parses = [p for p in analysis if not p.is_unknown()]
    if not parses:
        return TokenResult(token)

    parse, method = choose_parse(parses, best, ud_pos)
    n_case, n_ki = count_case_and_ki(parse)
    if mode == "postp" and is_postposition(parse):
        return TokenResult(
            stripped="",
            original_analysis=parse.format_string(),
            method=method,
            deleted_postp=parse.item.lemma,
            prefix=prefix,
            suffix=suffix,
        )
    if mode == "nominal":
        stripped, stripped_analysis, dropped = strip_nominal(core, parse)
    elif mode == "nominal_infl":
        stripped, stripped_analysis, dropped = strip_nominal_infl(core, parse)
    else:
        stripped, stripped_analysis, dropped = strip_cases(core, parse)
    return TokenResult(
        stripped=prefix + stripped + suffix if stripped else token,
        n_case=n_case,
        n_ki=n_ki,
        original_analysis=parse.format_string(),
        stripped_analysis=stripped_analysis,
        method=method,
        dropped=dropped,
    )


def disambiguate_cores(cores: list[str]) -> tuple[list, list]:
    """Zemberek analyses of each core plus the sentence-level best parse for each."""
    analyses = [zemb.morphology.analyze(core or ".") for core in cores]
    try:
        best = zemb.morphology.disambiguate(" ".join(cores), analyses).best_analysis()
    except Exception:  # noqa: BLE001 — fall back to POS + ranking only
        best = None
    if best is None or len(best) != len(cores):
        best = [None] * len(cores)
    best = [None if b is None or b.is_unknown() else b for b in best]
    return analyses, best


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


def strip_sentence(
    tagger, sentence: str, stats: CorpusStats, mode: str = "case"
) -> str:
    words = sentence.split()
    if not words:
        return sentence
    tags = pos_tag_words(tagger, words)
    analyses, best = disambiguate_cores([peel_punct(word)[1] for word in words])
    stripped_words: list[str] = []
    pending_prefix = ""  # leading punctuation of a deleted word ("(için" -> "(")
    for word, tag, analysis, best_parse in zip(words, tags, analyses, best):
        result = strip_token(word, tag, analysis, best_parse, mode)
        if result.method:
            stats.selection[result.method] += 1
        if result.deleted_postp is not None:
            # Drop the word, keep its punctuation on the neighbouring words:
            # "ev için, de" -> "ev, de".
            stats.n_tokens += 1
            stats.deleted_postp[result.deleted_postp] += 1
            pending_prefix += result.prefix
            if result.suffix:
                if stripped_words:
                    stripped_words[-1] += pending_prefix + result.suffix
                    pending_prefix = ""
                else:
                    pending_prefix += result.suffix
            continue
        if pending_prefix:
            result.stripped = pending_prefix + result.stripped
            pending_prefix = ""
        if result.stripped != word:
            stats.dropped_tags.update(result.dropped)
        stats.add_token(
            word,
            result.stripped,
            result.n_case,
            result.n_ki,
            result.original_analysis,
            result.stripped_analysis,
        )
        stripped_words.append(result.stripped)
    if pending_prefix:
        if stripped_words:
            stripped_words[-1] += pending_prefix
        else:
            stripped_words.append(pending_prefix)
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
    mode: str = "case",
) -> None:
    from tqdm import tqdm

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading TULAP POS tagger...", flush=True)
    tulap_tagger = load_tulap_tagger(device)
    print(f"Zemberek analyser ready. Mode: {mode}", flush=True)

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
            tqdm(in_f, total=n_lines, desc=f"Stripping ({mode})", unit="sent")
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
                out_f.write(strip_sentence(tulap_tagger, text, stats, mode) + "\n")
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
        description="Strip Turkish nominal suffixes from NLLB using TULAP POS + Zemberek",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--mode",
        choices=STRIP_MODES,
        default="case",
        help="case: every case suffix (incl. instrumental, equative) + -ki, "
        "keep plural and possessive; postp: case + delete postposition words; "
        "nominal: all nominal morphology, nominalised verbs back to the verb stem; "
        "nominal_infl: all nominal inflection, keep every derivation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: NLLB_10M_stripped{,_postp,_nominal,_nominal_infl}.tr by --mode",
    )
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
        output_path=args.output or DEFAULT_OUTPUTS[args.mode],
        device=args.device,
        max_lines=args.max_lines,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
