import re
import nltk
import zeyrek

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

analyzer = zeyrek.MorphAnalyzer()

TARGET_CASES = {"Dat", "Loc", "Abl", "Acc", "Gen"}

# Terminal unvoicing map to restore standard dictionary base
UNVOICING_MAP = {
    "b": "p",
    "c": "ç",
    "d": "t",
    "ğ": "k",
    "g": "k"
}

# Regex for stripping terminal case suffixes when intermediate morphemes (e.g. possessive) exist
CASE_SUFFIX_REGEX = {
    "Dat": re.compile(r"[yn]?[ae]$", re.IGNORECASE),
    "Loc": re.compile(r"[n]?[dt][ae]$", re.IGNORECASE),
    "Abl": re.compile(r"[n]?[dt][ae]n$", re.IGNORECASE),
    "Acc": re.compile(r"[yn]?[ıiuü]$", re.IGNORECASE),
    "Gen": re.compile(r"[yn]?[ıiuü]n$", re.IGNORECASE)
}

def unvoice_final_consonant(text: str) -> str:
    """Restores softened terminal consonants (b->p, c->ç, d->t, ğ->k)."""
    if text and text[-1] in UNVOICING_MAP:
        return text[:-1] + UNVOICING_MAP[text[-1]]
    return text

def find_best_case_parse(parses):
    """
    Selects the best parse, preferring valid noun parses with case suffixes
    over uninflected proper nouns or verb homonyms.
    """
    # 1. First priority: Noun parse that ends with a target case
    for p in parses:
        if p.pos in ("Noun", "Adj") and p.morphemes and p.morphemes[-1] in TARGET_CASES:
            # Skip if it's treated as a bare Proper Noun without case
            if "Prop" not in p.formatted or len(p.morphemes) > 1:
                return p

    # 2. Second priority: Any parse ending with a target case
    for p in parses:
        if p.morphemes and p.morphemes[-1] in TARGET_CASES:
            return p

    # 3. Fallback: Top ranked parse
    return parses[0]

def strip_case_and_restore_voicing(word: str) -> str:
    """
    Strips Dative, Locative, Ablative, or Accusative suffixes using Zeyrek.
    Properly handles unvoicing (yumuşama), vowel drops (hece düşmesi), 
    and noun/verb morphological ambiguity.
    """
    results = analyzer.analyze(word)
    if not results or not results[0]:
        return word

    parses = results[0]
    best_parse = find_best_case_parse(parses)
    print(f"Best parse: {best_parse}")
    morpheme_tags = best_parse.morphemes
    
    print(f"Morpheme tags: {morpheme_tags}")
    if not morpheme_tags:
        return word

    last_tag = morpheme_tags[-1]
    print(f"Last tag: {last_tag}")

    # Check if the chosen parse contains a target case suffix
    if last_tag in TARGET_CASES:
        has_intermediate = any(
            tag in morpheme_tags 
            for tag in ("A3pl", "P1sg", "P2sg", "P3sg", "P1pl", "P2pl", "P3pl")
        )
        print(f"Has intermediate: {has_intermediate}")
        # Case A: Bare Root + Case -> Return canonical unvoiced dictionary lemma
        if not has_intermediate:
            lemma = best_parse.lemma
            
            # Extract root directly from formatted string if lemma retained softened form
            # e.g., formatted '<(kitap_Noun)...>' -> 'kitap'
            match = re.search(r"\(([a-zA-ZçğıöşüÇĞİÖŞÜ]+)_(?:Noun|Adj|Prop)", best_parse.formatted)
            if match:
                lemma = match.group(1)

            # Apply final unvoicing safeguard (e.g. kitab -> kitap, kanad -> kanat)
            lemma = unvoice_final_consonant(lemma.lower())

            if word.istitle():
                return lemma.capitalize()
            elif word.isupper():
                return lemma.upper()
            return lemma
        print(f"No intermediate suffixes found for {word}")

        # Case B: Intermediate suffixes exist (e.g. kitab + ım + ı -> kitabım)
        pattern = CASE_SUFFIX_REGEX.get(last_tag)
        print(f"Pattern: {pattern}")
        if pattern:
            return pattern.sub("", word)
        print(f"No pattern found for {word}")
    return word
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
