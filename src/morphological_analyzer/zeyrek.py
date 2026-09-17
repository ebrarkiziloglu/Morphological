"""
Zeyrek morphological analyser helpers.

Uses same case extraction as Zemberek (format compatible).
"""

from typing import List, Optional, Tuple

# Cached analyzer instance
_ZEYREK_ANALYZER = None

from .zemberek import get_case_prediction_zemberek


def run_zeyrek_analyser(
    words: List[str],
    sentence_text: Optional[str] = None,
) -> Tuple[dict, set]:
    """Run Zeyrek morphological analyser. Returns analyses as formatted strings per word."""
    global _ZEYREK_ANALYZER

    try:
        import logging
        import zeyrek
        logging.getLogger('zeyrek').setLevel(logging.ERROR)
    except ImportError:
        print("Error: zeyrek not installed. Install with: pip install zeyrek")
        return {}, set(words) if words else set()

    if not words:
        return {}, set()

    if _ZEYREK_ANALYZER is None:
        _ZEYREK_ANALYZER = zeyrek.MorphAnalyzer()
    analyzer = _ZEYREK_ANALYZER
    analyses_dict: dict = {}
    no_result_words: set = set()

    for w in words:
        try:
            word_analyses = analyzer.analyze(w)
            if word_analyses and len(word_analyses) > 0:
                parses = word_analyses[0]
                if parses:
                    analyses_dict[w] = [p.formatted for p in parses]
                else:
                    no_result_words.add(w)
            else:
                no_result_words.add(w)
        except Exception:
            no_result_words.add(w)

    return analyses_dict, no_result_words


def get_case_prediction_zeyrek(analyses: List[str]) -> Optional[str]:
    """Predict UD Case from Zeyrek analyses (same format as Zemberek)."""
    return get_case_prediction_zemberek(analyses)
