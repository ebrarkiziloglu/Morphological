"""
Strip Turkish case suffixes, preferring noun analyses.

Same logic as remove_case_suffixes_from_tr_prioritize_nouns.py, but uses
Zemberek (zemberek-python) the same way morphological_analyzer/zemberek.py does.
"""

import logging
import re
import sys
import typing

# zemberek-python expects the removed typing.io module on some Python versions
typing.io = typing  # type: ignore[attr-assign]
sys.modules.setdefault("typing.io", typing)

logging.getLogger("zemberek").setLevel(logging.WARNING)

from zemberek import TurkishMorphology

morphology = TurkishMorphology.create_with_defaults()

TARGET_CASES = {"Dat", "Loc", "Abl", "Acc", "Gen"}
INTERMEDIATE_TAGS = {"A3pl", "P1sg", "P2sg", "P3sg", "P1pl", "P2pl", "P3pl"}
NOMINAL_POS = {"Noun", "Adj", "Pron"}
SKIP_SECONDARY_POS = {"Prop", "Abbrv"}
POS_TAGS = {
    "Noun", "Adj", "Verb", "Pron", "Adv", "Num", "Postp",
    "Conj", "Det", "Ques", "Interj", "Punc",
}

# Terminal unvoicing map to restore standard dictionary base
UNVOICING_MAP = {
    "b": "p",
    "c": "ç",
    "d": "t",
    "ğ": "k",
    "g": "k",
}

# Regex for stripping terminal case suffixes when intermediate morphemes exist
CASE_SUFFIX_REGEX = {
    "Dat": re.compile(r"[yn]?[ae]$", re.IGNORECASE),
    "Loc": re.compile(r"[n]?[dt][ae]$", re.IGNORECASE),
    "Abl": re.compile(r"[n]?[dt][ae]n$", re.IGNORECASE),
    "Acc": re.compile(r"[yn]?[ıiuü]$", re.IGNORECASE),
    "Gen": re.compile(r"[yn]?[ıiuü]n$", re.IGNORECASE),
}


def unvoice_final_consonant(text: str) -> str:
    """Restores softened terminal consonants (b->p, c->ç, d->t, ğ->k)."""
    if text and text[-1] in UNVOICING_MAP:
        return text[:-1] + UNVOICING_MAP[text[-1]]
    return text


def morpheme_ids(parse) -> list:
    return [m.morpheme.id_ for m in parse.morpheme_data_list]


def last_pos_tag(tags) -> str | None:
    """POS of the last inflectional group (handles derivations such as Verb|Agt→Noun)."""
    for tag in reversed(tags):
        if tag in POS_TAGS:
            return tag
    return None


def parse_preference_score(parse) -> tuple:
    """Tie-break among case parses: pronoun > noun/adj, then P3sg, then longer case surface."""
    tags = morpheme_ids(parse)
    last_pos = last_pos_tag(tags)
    last_surface = parse.morpheme_data_list[-1].surface if parse.morpheme_data_list else ""
    if last_pos == "Pron":
        pos_rank = 2
    elif last_pos in {"Noun", "Adj"}:
        pos_rank = 1
    else:
        pos_rank = 0
    p3sg = 1 if "P3sg" in tags else 0
    return (pos_rank, p3sg, len(last_surface))


def find_best_case_parse(parses):
    # TODO: Use the POS-tag of the word to find the best parse
    return parses[0]

    """
    Selects the best parse, preferring valid noun/pronoun parses with case suffixes
    over uninflected proper nouns or verb homonyms.
    """
    def ends_with_target_case(parse) -> bool:
        tags = morpheme_ids(parse)
        return bool(tags) and tags[-1] in TARGET_CASES

    # 1. First priority: common Noun/Adj/Pron parse that ends with a target case
    nominal_case_parses = []
    for p in parses:
        tags = morpheme_ids(p)
        spos = p.item.secondary_pos.short_form
        if (
            last_pos_tag(tags) in NOMINAL_POS
            and ends_with_target_case(p)
            and spos not in SKIP_SECONDARY_POS
        ):
            nominal_case_parses.append(p)
    if nominal_case_parses:
        return max(nominal_case_parses, key=parse_preference_score)

    # 2. Second priority: any parse ending with a target case
    any_case_parses = [p for p in parses if ends_with_target_case(p)]
    if any_case_parses:
        return max(any_case_parses, key=parse_preference_score)

    # 3. Fallback: top ranked parse
    return parses[0]


def restore_word_case(original: str, stripped: str) -> str:
    if original.istitle():
        return stripped.capitalize()
    if original.isupper():
        return stripped.upper()
    return stripped


def strip_case_suffix_from_surface(word: str, last_tag: str, last_surface: str) -> str:
    """Drop the case suffix using Zemberek's surface, then the regex fallback."""
    if last_surface and word.lower().endswith(last_surface.lower()):
        return word[: -len(last_surface)]
    pattern = CASE_SUFFIX_REGEX.get(last_tag)
    if pattern:
        return pattern.sub("", word)
    return word


def strip_case_and_restore_voicing(word: str) -> str:
    """
    Strips Dative, Locative, Ablative, Accusative, or Genitive suffixes using Zemberek.
    Properly handles unvoicing (yumuşama), vowel drops (hece düşmesi),
    and noun/verb morphological ambiguity.
    """
    analysis = morphology.analyze(word)
    parses = [p for p in analysis if not p.is_unknown()]
    if not parses:
        return word

    best_parse = find_best_case_parse(parses)
    print(f"Best parse: {best_parse.format_string()}")
    tags = morpheme_ids(best_parse)
    print(f"Morpheme tags: {tags}")
    if not tags:
        return word

    last_tag = tags[-1]
    print(f"Last tag: {last_tag}")

    if last_tag not in TARGET_CASES:
        return word

    has_intermediate = any(tag in INTERMEDIATE_TAGS for tag in tags)
    print(f"Has intermediate: {has_intermediate}")

    # Case A: Bare Root + Case -> canonical unvoiced dictionary lemma
    if not has_intermediate:
        lemma = best_parse.item.lemma
        # Keep dictionary lemma when present; otherwise fall back to stem in the format string
        # e.g. formatted '[kitap:Noun] kitab:Noun+A3sg+ı:Acc' -> 'kitap'
        match = re.search(
            r"\[([a-zA-ZçğıöşüÇĞİÖŞÜ]+):(?:Noun|Adj|Pron)",
            best_parse.format_string(),
        )
        if match:
            lemma = match.group(1)

        lemma = unvoice_final_consonant(lemma.lower())
        return restore_word_case(word, lemma)

    # Case B: Intermediate suffixes exist (e.g. kitab + ım + ı -> kitabım)
    last_surface = best_parse.morpheme_data_list[-1].surface
    print(f"Last surface: {last_surface!r}")
    return strip_case_suffix_from_surface(word, last_tag, last_surface)


# Test cases
test_words = [
    # 1. Bare Root + Case (restores unvoicing & vowel drop)
    "kitabı",       # Accusative -> kitap
    "ağaca",        # Dative     -> ağaç
    "kanadı",       # Accusative -> kanat
    "köpeğe",       # Dative     -> köpek
    "burnu",        # Accusative -> burun (vowel drop: burun + u -> burnu)
    "şehre",        # Dative     -> şehir (vowel drop: şehir + e -> şehre)

    # 2. Root + Possessive + Case (keeps modified stem + possessive)
    "kitabımı",     # Accusative -> kitabım
    "ağacında",     # Locative   -> ağacı
    "köpeğime",     # Dative     -> köpeğim
    "burnumu",      # Accusative -> burnum
    "şehrinde",     # Locative   -> şehri

    # 3. Regular cases / plurals (no root mutation)
    "masadan",      # Ablative   -> masa
    "okula",        # Dative     -> okul
    "evlerimde",    # Locative   -> evlerim
    "öğretmenler",  # Plural, no case -> öğretmenler
    "kullanıcılarının",
    "benim",
    "sana",         # sen + Dat, stem e->a
    "senin",
    "onun",
    "onların",
    "bizim",

    # 4. p -> b voicing (yumuşama); Loc/Abl keep p because the suffix starts with a consonant
    "kitaba",       # Dat -> kitap
    "dolabı",       # Acc -> dolap
    "sebebi",       # Acc -> sebep
    "cevabı",       # Acc -> cevap
    "hesabı",       # Acc -> hesap
    "şarabı",       # Acc -> şarap
    "mektubu",      # Acc -> mektup
    "kalıbı",       # Acc -> kalıp
    "cebi",         # Acc -> cep
    "kitapta",      # Loc, p stays -> kitap
    "kitaptan",     # Abl, p stays -> kitap

    # 5. ç -> c voicing
    "ağacı",        # Acc -> ağaç
    "amacı",        # Acc -> amaç
    "sonucu",       # Acc -> sonuç
    "aracı",        # Acc -> araç
    "kılıcı",       # Acc -> kılıç
    "ilacı",        # Acc -> ilâç (Zemberek lemma uses â)
    "inancı",       # Acc -> inanç
    "kazancı",      # Acc -> kazanç
    "borcu",        # Acc -> borç
    "genci",        # Acc -> genç
    "ağaçta",       # Loc, ç stays -> ağaç
    "ağaçtan",      # Abl, ç stays -> ağaç

    # 6. t -> d voicing vs t that stays (many loans)
    "umudu",        # Acc -> umut
    "yoğurdu",      # Acc -> yoğurt
    "yurdu",        # Acc -> yurt
    "dördü",        # Acc -> dört
    "derdi",        # Acc -> dert
    "kilidi",       # Acc -> kilit
    "senedi",       # Acc -> senet
    "adedi",        # Acc -> adet
    "vücudu",       # Acc -> vücut
    "muradı",       # Acc -> murat
    "kanatta",      # Loc, t stays -> kanat
    "hayatı",       # Acc, t stays -> hayat
    "saati",        # Acc, t stays -> saat

    # 7. k -> ğ voicing
    "bebeği",       # Acc -> bebek
    "ayağa",        # Dat -> ayak
    "sokağı",       # Acc -> sokak
    "ekmeği",       # Acc -> ekmek
    "ekmeğe",       # Dat -> ekmek
    "ineği",        # Acc -> inek
    "çiçeği",       # Acc -> çiçek
    "bileği",       # Acc -> bilek
    "yüreği",       # Acc -> yürek
    "parmağı",      # Acc -> parmak
    "bacağı",       # Acc -> bacak
    "yatağı",       # Acc -> yatak
    "ocağı",        # Acc -> ocak
    "uçağı",        # Acc -> uçak
    "köpekte",      # Loc, k stays -> köpek

    # 8. k -> g after n (renk, ahenk, ...)
    "rengi",        # Acc -> renk
    "renge",        # Dat -> renk
    "ahengi",       # Acc -> ahenk
    "çelengi",      # Acc -> çelenk
    "kepengi",      # Acc -> kepenk

    # 9. Vowel drop / hece düşmesi (second vowel deleted: burun+u -> burnu)
    "burna",        # Dat -> burun
    "burnun",       # Gen -> burun
    "şehri",        # Acc -> şehir
    "şehrin",       # Gen -> şehir
    "ağzı",         # Acc -> ağız
    "ağza",         # Dat -> ağız
    "oğlu",         # Acc -> oğul
    "oğla",         # Dat -> oğul
    "boynu",        # Acc -> boyun
    "boyna",        # Dat -> boyun
    "karnı",        # Acc -> karın
    "karna",        # Dat -> karın
    "alnı",         # Acc -> alın
    "omzu",         # Acc -> omuz
    "beyni",        # Acc -> beyin
    "beyne",        # Dat -> beyin
    "göğsü",        # Acc -> göğüs
    "nehri",        # Acc -> nehir
    "nehre",        # Dat -> nehir
    "fikri",        # Acc -> fikir
    "ömrü",         # Acc -> ömür
    "ismi",         # Acc -> isim
    "isme",         # Dat -> isim
    "resmi",        # Acc -> resim
    "keşfi",        # Acc -> keşif
    "vakti",        # Acc -> vakit
    "nakli",        # Acc -> nakil
    "aklı",         # Acc -> akıl
    "sabrı",        # Acc -> sabır
    "izni",         # Acc -> izin
    "keyfi",        # Acc -> keyif
    "metni",        # Acc -> metin
    "hapsi",        # Acc -> hapis
    "bağrı",        # Acc -> bağır
    "koynu",        # Acc -> koyun (bosom; contrast sheep: koyunu)

    # 10. Vowel drop AND voicing together
    "kaybı",        # Acc -> kayıp (ı drops, p->b)
    "kayba",        # Dat -> kayıp

    # 11. Possessive + case on already mutated stems
    "ağzından",     # Abl -> ağzı   (ağız + P3sg + Abl)
    "oğluna",       # Dat -> oğlu   (oğul + P3sg + Dat)
    "beynime",      # Dat -> beynim (beyin + P1sg + Dat)
    "karnında",     # Loc -> karnı  (karın + P3sg + Loc)
    "rengine",      # Dat -> rengi  (renk + P3sg + Dat)
    "ayağımda",     # Loc -> ayağım (ayak + P1sg + Loc)
    "burnunda",     # Loc -> burnu  (burun + P3sg + Loc)
    "şehrine",      # Dat -> şehri  (şehir + P3sg + Dat)

    # 12. Buffer y (vowel-final stem, not a root mutation)
    "suyu",         # Acc -> su
    "suya",         # Dat -> su
    "kutuya",       # Dat -> kutu
    "kapıya",       # Dat -> kapı
    "arabaya",      # Dat -> araba
    "suyun",        # Gen -> su

    # 13. Pronoun stem change (ben/sen e->a)
    "bana",         # Dat -> ben
    "bende",        # Loc -> ben
    "benden",       # Abl -> ben
    "bizi",         # Acc -> biz
    "bize",         # Dat -> biz
    "size",         # Dat -> siz
    "ona",          # Dat -> o
    "buna",         # Dat -> bu
    "şuna",         # Dat -> şu

    # 14. Monosyllables that usually do NOT voice
    "atı",          # Acc -> at   (t stays)
    "saçı",         # Acc -> saç  (ç stays)
    "topu",         # Acc -> top  (p stays)
    "ipi",          # Acc -> ip   (p stays)
    "suçu",         # Acc -> suç  (ç stays)
    "gücü",         # Acc -> güç  (ç stays)
    "koyunları",    # Acc -> koyunlar (sheep plural; no vowel drop, contrast koynu)
]


if __name__ == "__main__":
    print(f"{'Original':<22} -> {'Stripped'}")
    print("-" * 40)
    final_output = ""
    for w in test_words:
        stripped = strip_case_and_restore_voicing(w)
        print(f"{w:<22} -> {stripped}")
        final_output += f"{w:<22} -> {stripped}\n"
    print("\n", "-" * 40, "\n")
    print(final_output)


# with open("/home/hpc/b279bb/b279bb15/Morphological/data/NLLB_10M_subset/NLLB_small.tr", "r") as f:
#     data = f.readlines()

# with open("/home/hpc/b279bb/b279bb15/Morphological/data/NLLB_10M_subset/NLLB_small_stripped.tr", "w") as f:
#     for line in data:
#         for word in line.split():
#             stripped_word = strip_case_and_restore_voicing(word)
#             f.write(stripped_word + " ")
#         f.write("\n")