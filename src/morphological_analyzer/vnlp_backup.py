"""
vnlp StemmerAnalyzer morphological analyser helpers.
"""

from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

VNLP_UD_CASES = ('Nom', 'Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins')

from .trmor import CASE_PRIORITY

_STEMMER = None


def get_stemmer() -> Optional[Any]:
    """Lazy loader for the StemmerAnalyzer. Returns None if vnlp is not installed."""
    global _STEMMER
    if _STEMMER is None:
        try:
            from vnlp import StemmerAnalyzer
            _STEMMER = StemmerAnalyzer()
        except ImportError:
            print("Error: vnlp not installed. Install with: pip install vnlp")
            return None
    return _STEMMER


@lru_cache(maxsize=10000)
def cached_predict(sentence: str, batch_size: int = 256) -> tuple:
    """Cached predict for repeated identical sentences. Returns tuple of analyses (empty if stemmer unavailable)."""
    stemmer = get_stemmer()
    if stemmer is None:
        return ()
    return tuple(stemmer.predict(sentence, batch_size=batch_size))


def extract_case_from_vnlp(analysis: str) -> Optional[str]:
    """Extract UD Case from vnlp StemmerAnalyzer output (e.g. 'üniversite+Noun+A3sg+Pnon+Nom')."""
    if not analysis or '+' not in analysis:
        return None
    parts = analysis.split('+')
    if parts and parts[-1] in VNLP_UD_CASES:
        return parts[-1]
    return None


def run_vnlp_analyser(
    words: List[str],
    sentence_tokens: Optional[List[Dict[str, Any]]] = None,
    sentence_text: Optional[str] = None,
    batch_size: int = 256,
) -> Tuple[dict, set]:
    """Run vnlp StemmerAnalyzer. Uses cached_predict for repeated sentences. When sentence_text
    and sentence_tokens are given, aligns predictions by position. Otherwise runs one batched
    cached predict on space-joined words (lemma path). Use batch_size for GPU tuning."""
    from morphological_analysis_helpers import turkish_lower

    analyses_dict = defaultdict(list)
    no_result_words = set()
    words_set = set(turkish_lower(w) for w in words) if words else set()

    if not words and not (sentence_tokens and sentence_text):
        return {}, set()

    if get_stemmer() is None:
        return {}, words_set

    try:
        if sentence_text and sentence_tokens:
            pred = cached_predict(sentence_text, batch_size)
            if len(pred) != len(sentence_tokens):
                return {}, words_set
            for i, token in enumerate(sentence_tokens):
                key = turkish_lower(token['form'])
                analyses_dict[key].append(pred[i])

        elif words:
            reconstructed_sentence = " ".join(words)
            pred = cached_predict(reconstructed_sentence, batch_size)
            if pred and len(pred) == len(words):
                for i, w in enumerate(words):
                    analyses_dict[turkish_lower(w)].append(pred[i])
            else:
                for w in words:
                    no_result_words.add(turkish_lower(w))

        for w in words:
            key = turkish_lower(w)
            if key not in analyses_dict and key not in no_result_words:
                no_result_words.add(key)

        return dict(analyses_dict), no_result_words
    except Exception as e:
        print(
            "Error: vnlp StemmerAnalyzer failed due to network/model issues. "
            "This often happens on clusters without internet access for downloading "
            "VNLP model weights (Stemmer_Shen_prod.weights). "
            f"Details: {e}"
        )
        return {}, words_set


def get_case_prediction_vnlp(analyses: List[str]) -> Optional[str]:
    """Predict UD Case from vnlp analyses (majority vote)."""
    if not analyses:
        return None
    case_counts = defaultdict(int)
    for analysis in analyses:
        case = extract_case_from_vnlp(analysis)
        if case:
            case_counts[case] += 1
    if not case_counts:
        return 'Nom'
    max_count = max(case_counts.values())
    candidates = [c for c, cnt in case_counts.items() if cnt == max_count]
    for case in CASE_PRIORITY:
        if case in candidates:
            return case
    return candidates[0] if candidates else 'Nom'
