"""Heuristic alignment of German dative NP spans to Turkish in parallel UD PUD."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

DAT_POS: Set[str] = {"NOUN", "PROPN", "PRON"}
GERMAN_SPAN_DEPRELS: Set[str] = {
    "det",
    "amod",
    "nummod",
    "compound",
    "fixed",
    "flat",
}
TURKISH_SPAN_DEPRELS: Set[str] = {
    *GERMAN_SPAN_DEPRELS,
    "nmod",
    "nmod:poss",
    "case",
    "advmod:emph",
}

# German dative preposition → typical Turkish UD Case values.
PREP_TO_TR_CASES: Dict[str, Set[str]] = {
    "in": {"Loc", "Dat"},
    "zu": {"Dat"},
    "mit": {"Ins", "Com", "Nom"},
    "von": {"Abl", "Gen"},
    "an": {"Loc", "Dat"},
    "nach": {"Dat"},
    "bei": {"Loc"},
    "aus": {"Abl"},
    "auf": {"Loc", "Dat"},
    "vor": {"Abl", "Loc"},
    "unter": {"Loc"},
    "seit": {"Abl"},
}

GERMAN_WEEKDAY_TO_TURKISH: Dict[str, str] = {
    "montag": "pazartesi",
    "dienstag": "salı",
    "mittwoch": "çarşamba",
    "donnerstag": "perşembe",
    "freitag": "cuma",
    "samstag": "cumartesi",
    "sonntag": "pazar",
}

# Lemma-level translation hints for frequent dative heads without shared form.
GERMAN_LEMMA_HINTS: Dict[str, Set[str]] = {
    "hand": {"el", "eliniz", "eller"},
    "wohnzimmer": {"oturma", "oda", "odası", "odasında"},
    "menschenmenge": {"kalabalık", "kalabalığa"},
    "vergleich": {"karşılaştır", "karşılaştırma"},
    "minute": {"dakika"},
    "wettbewerb": {"rekabet"},
    "blogeintrag": {"yazı", "blog"},
    "tochter": {"kız", "kızı"},
    "sportplatz": {"saha", "sahada"},
    "universität": {"üniversite"},
    "university": {"üniversite"},
    "gondel": {"telesiyej"},
    "club": {"club"},
}
GERMAN_MONTH_TO_TURKISH: Dict[str, str] = {
    "januar": "ocak",
    "februar": "şubat",
    "märz": "mart",
    "april": "nisan",
    "mai": "mayıs",
    "juni": "haziran",
    "juli": "temmuz",
    "august": "ağustos",
    "september": "eylül",
    "oktober": "ekim",
    "november": "kasım",
    "dezember": "aralık",
}

_SUFFIX_RE = re.compile(
    r"(?:"
    r"'(?:da|de|ta|te|nda|nde|ndan|nden|tan|ten|dan|den|"
    r"a|e|ya|ye|na|ne|la|le|yla|yle)"
    r")(?:[^a-zA-Z]|$)",
    re.IGNORECASE,
)


def collect_span_tokens(
    tokens: List[Dict[str, Any]],
    head_id: str,
    *,
    extra: Set[str] = GERMAN_SPAN_DEPRELS,
) -> List[Dict[str, Any]]:
    span = [
        t
        for t in tokens
        if t["id"] == head_id
        or (t["head"] == head_id and t["deprel"] in extra)
    ]
    span.sort(key=lambda t: int(str(t["id"]).split("-")[0]))
    return span


def collect_span_text(tokens: List[Dict[str, Any]], head_id: str) -> str:
    return " ".join(t["form"] for t in collect_span_tokens(tokens, head_id))


def turkish_case(token: Dict[str, Any]) -> Optional[str]:
    case = token["feats"].get("Case")
    if case:
        return case
    match = _SUFFIX_RE.search(token["form"] + " ")
    if not match:
        return None
    suffix = match.group(0).lower().strip("'")
    if suffix in {"da", "de", "ta", "te", "nda", "nde"}:
        return "Loc"
    if suffix in {"dan", "den", "tan", "ten", "ndan", "nden"}:
        return "Abl"
    if suffix in {"a", "e", "ya", "ye", "na", "ne"}:
        return "Dat"
    if suffix in {"la", "le", "yla", "yle"}:
        return "Ins"
    return None


def _token_by_id(tokens: List[Dict[str, Any]], token_id: str) -> Optional[Dict[str, Any]]:
    return next((t for t in tokens if t["id"] == token_id), None)


_GERMAN_STOPWORDS = {
    "der",
    "die",
    "das",
    "dem",
    "den",
    "des",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
}


def _anchors_from_german_span(de_span: List[Dict[str, Any]]) -> Set[str]:
    anchors: Set[str] = set()
    for token in de_span:
        if token["upos"] == "DET":
            continue
        for value in (token["form"], token["lemma"]):
            cleaned = value.strip()
            if cleaned.lower() in _GERMAN_STOPWORDS:
                continue
            if len(cleaned) >= 2:
                anchors.add(cleaned.lower())
            if cleaned[:1].isupper() or token["upos"] == "PROPN":
                anchors.add(cleaned)
        month = GERMAN_MONTH_TO_TURKISH.get(token["lemma"].lower())
        if month:
            anchors.add(month)
            anchors.add(month.capitalize())
        weekday = GERMAN_WEEKDAY_TO_TURKISH.get(token["lemma"].lower())
        if weekday:
            anchors.add(weekday)
            anchors.add(weekday.capitalize())
    return anchors


def _lemma_hint_overlap(de_head: Dict[str, Any], token: Dict[str, Any]) -> int:
    hints = GERMAN_LEMMA_HINTS.get(de_head["lemma"].lower(), set())
    if not hints:
        return 0
    haystacks = {token["lemma"].lower(), token["form"].lower()}
    for hint in hints:
        for hay in haystacks:
            if hint in hay:
                return 35
    return 0


def _anchor_overlap_score(anchors: Set[str], window: List[Dict[str, Any]]) -> int:
    score = 0
    for token in window:
        for value in (token["form"], token["lemma"]):
            lower = value.lower()
            upper = value
            for anchor in anchors:
                anchor_lower = anchor.lower()
                if anchor_lower == lower:
                    score += 20
                elif len(anchor_lower) >= 4 and (
                    anchor_lower in lower or lower in anchor_lower
                ):
                    score += 12
                elif anchor == upper:
                    score += 20
    return score


def _collect_turkish_span(tokens: List[Dict[str, Any]], head_id: str) -> str:
    span_ids: Set[str] = {head_id}
    changed = True
    while changed:
        changed = False
        for token in tokens:
            if token["id"] in span_ids:
                continue
            if token["head"] in span_ids and token["deprel"] in TURKISH_SPAN_DEPRELS:
                span_ids.add(token["id"])
                changed = True
        for token in tokens:
            if token["id"] in span_ids:
                continue
            if (
                token["head"] in span_ids
                and token["deprel"] in {"nmod", "nmod:poss"}
                and token["upos"] in DAT_POS | {"ADJ", "DET", "PROPN"}
            ):
                span_ids.add(token["id"])
                changed = True
    span = [token for token in tokens if token["id"] in span_ids]
    span.sort(key=lambda t: int(str(t["id"]).split("-")[0]))
    return " ".join(token["form"] for token in span)


def _score_turkish_candidate(
    *,
    prep: str,
    de_span: List[Dict[str, Any]],
    de_tokens: List[Dict[str, Any]],
    de_head_id: str,
    tr_tokens: List[Dict[str, Any]],
    tr_idx: int,
) -> int:
    token = tr_tokens[tr_idx]
    expected_cases = PREP_TO_TR_CASES.get(prep, set())
    case = turkish_case(token)
    score = 0

    if case in expected_cases:
        score += 15
    elif prep == "in" and case == "Dat":
        score += 8
    elif case:
        score -= 8

    deprel = token["deprel"].split(":")[0]
    if deprel in {"obl", "iobj", "obj", "nmod", "conj", "advcl"}:
        score += 2

    anchors = _anchors_from_german_span(de_span)
    window = tr_tokens[max(0, tr_idx - 4) : tr_idx + 5]
    score += _anchor_overlap_score(anchors, window)

    de_head = _token_by_id(de_tokens, de_head_id)
    if de_head:
        month = GERMAN_MONTH_TO_TURKISH.get(de_head["lemma"].lower())
        if month and month in token["lemma"].lower():
            score += 35
        weekday = GERMAN_WEEKDAY_TO_TURKISH.get(de_head["lemma"].lower())
        if weekday and weekday in token["lemma"].lower():
            score += 35
        score += _lemma_hint_overlap(de_head, token)

    de_numbers = {t["form"] for t in de_span if any(ch.isdigit() for ch in t["form"])}
    for window_token in window:
        if window_token["form"] in de_numbers:
            score += 25

    governor = _token_by_id(de_tokens, de_head_id)
    if governor and _token_by_id(de_tokens, governor["head"]) is not None:
        de_governor = _token_by_id(de_tokens, governor["head"])
        if de_governor and de_governor["upos"] == "VERB":
            for window_token in window:
                head = _token_by_id(tr_tokens, window_token["head"])
                if head and head["upos"] == "VERB":
                    score += 3

    return score


def find_turkish_dative_span(
    de_sentence: Dict[str, Any],
    tr_sentence: Dict[str, Any],
    *,
    de_head_id: str,
    prep: str,
    min_score: int = 6,
) -> Tuple[Optional[str], int, Optional[str]]:
    """Return ``(turkish_group_span_text, score, turkish_head_form)``."""
    de_tokens = de_sentence["tokens"]
    tr_tokens = tr_sentence["tokens"]
    de_span = collect_span_tokens(de_tokens, de_head_id)

    candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, token in enumerate(tr_tokens):
        if token["upos"] not in DAT_POS:
            continue
        score = _score_turkish_candidate(
            prep=prep,
            de_span=de_span,
            de_tokens=de_tokens,
            de_head_id=de_head_id,
            tr_tokens=tr_tokens,
            tr_idx=idx,
        )
        if score > 0:
            candidates.append((score, idx, token))

    if not candidates:
        for idx, token in enumerate(tr_tokens):
            if token["upos"] not in DAT_POS:
                continue
            if token["deprel"].split(":")[0] not in {
                "obl",
                "iobj",
                "obj",
                "nmod",
                "conj",
            }:
                continue
            score = _score_turkish_candidate(
                prep=prep,
                de_span=de_span,
                de_tokens=de_tokens,
                de_head_id=de_head_id,
                tr_tokens=tr_tokens,
                tr_idx=idx,
            )
            candidates.append((score, idx, token))

    if not candidates:
        return None, 0, None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_token = candidates[0]
    if best_score < min_score:
        return None, best_score, best_token["form"]

    span_text = _collect_turkish_span(tr_tokens, best_token["id"])
    return span_text, best_score, best_token["form"]
