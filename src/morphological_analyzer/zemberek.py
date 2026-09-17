"""
Zemberek morphological analyser helpers.
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

# Cached analyzer instance (used when not using ProcessPoolExecutor)
_ZEMBEREK_MORPHOLOGY = None

ZEMBEREK_CASE_TAGS = ('Gen', 'Dat', 'Acc', 'Loc', 'Abl', 'Ins')

# CASE_PRIORITY for tie-breaking in get_case_prediction_zemberek
from .trmor import CASE_PRIORITY


def extract_case_from_zemberek(analysis: str) -> Optional[str]:
    """Extract UD Case from Zemberek analysis string."""
    last_pos = -1
    last_case = None
    for case in ZEMBEREK_CASE_TAGS:
        tag = ':' + case
        pos = analysis.rfind(tag)
        if pos != -1:
            after = analysis[pos + len(tag):]
            if not after or after.startswith('+'):
                if pos > last_pos:
                    last_pos = pos
                    last_case = case
    if last_case:
        return last_case
    
    # Only default to 'Nom' if it actually acts nominally (Noun, Pronoun, Adjective)
    if ':Noun' in analysis or ':Pron' in analysis:
        return 'Nom'
        
    return None


def _analyze_with_morphology(
    morphology,
    words: List[str],
    sentence_text: Optional[str],
    use_disambiguation: bool,
) -> Tuple[dict, set]:
    """Run analysis using an existing TurkishMorphology instance."""
    analyses_dict: dict = {}
    no_result_words: set = set()

    if sentence_text and use_disambiguation:
        try:
            analysis = morphology.analyze_sentence(sentence_text)
            disambiguated = morphology.disambiguate(sentence_text, analysis)
            best = list(disambiguated.best_analysis())
            if len(best) == len(words):
                for i, word in enumerate(words):
                    parse = best[i]
                    if parse:
                        analyses_dict[word] = [parse.format_string()]
                    else:
                        no_result_words.add(word)
            else:
                for w in words:
                    results = morphology.analyze(w)
                    if results:
                        analyses_dict[w] = [r.format_string() for r in results]
                    else:
                        no_result_words.add(w)
        except Exception:
            for w in words:
                results = morphology.analyze(w)
                if results:
                    analyses_dict[w] = [r.format_string() for r in results]
                else:
                    no_result_words.add(w)
    else:
        for w in words:
            results = morphology.analyze(w)
            if results:
                analyses_dict[w] = [r.format_string() for r in results]
            else:
                no_result_words.add(w)

    return analyses_dict, no_result_words


def _worker_zemberek(args: Tuple[List[str], Optional[str], bool]) -> Tuple[dict, set]:
    """
    Picklable worker for ProcessPoolExecutor. Creates its own TurkishMorphology
    and runs analysis on the given (words, sentence_text, use_disambiguation).
    """
    words, sentence_text, use_disambiguation = args
    if not words:
        return {}, set()
    try:
        import logging
        logging.getLogger('zemberek').setLevel(logging.WARNING)
        import sys
        import typing
        typing.io = typing  # type: ignore[attr-assign]
        sys.modules.setdefault("typing.io", typing)

        from zemberek import TurkishMorphology

        morphology = TurkishMorphology.create_with_defaults()
        return _analyze_with_morphology(morphology, words, sentence_text, use_disambiguation)
    except ImportError:
        return {}, set(words)
    except Exception:
        return {}, set(words)


def run_zemberek_analyser(
    words: List[str],
    sentence_text: Optional[str] = None,
    use_disambiguation: bool = True,
    max_workers: Optional[int] = None,
) -> Tuple[dict, set]:
    """Run Zemberek morphological analyser.

    When max_workers > 1, uses ProcessPoolExecutor for parallelism (each process
    loads its own model). With sentence-level disambiguation only one task runs;
    without it, words are chunked across workers for maximum throughput.
    """
    global _ZEMBEREK_MORPHOLOGY

    try:
        import logging
        logging.getLogger('zemberek').setLevel(logging.WARNING)
        import sys
        import typing
        typing.io = typing  # type: ignore[attr-assign]
        sys.modules.setdefault("typing.io", typing)

        from zemberek import TurkishMorphology
    except ImportError as e:
        print(f"Import Error in zemberek-python: {e}")
        return {}, set(words)

    if not words:
        return {}, set()

    n_workers = max_workers if max_workers is not None else 0
    if n_workers > 1:
        # Parallel path: chunk words when no sentence disambiguation, else one task
        if sentence_text and use_disambiguation:
            tasks = [(words, sentence_text, use_disambiguation)]
        else:
            n_chunks = min(n_workers, len(words))
            chunk_size = (len(words) + n_chunks - 1) // n_chunks
            tasks = [
                (words[i : i + chunk_size], None, False)
                for i in range(0, len(words), chunk_size)
            ]
        merged_dict: dict = {}
        merged_no_result: set = set()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_zemberek, t): t for t in tasks}
            for fut in as_completed(futures):
                d, no_res = fut.result()
                merged_dict.update(d)
                merged_no_result |= no_res
        return merged_dict, merged_no_result

    # Sequential path
    if _ZEMBEREK_MORPHOLOGY is None:
        _ZEMBEREK_MORPHOLOGY = TurkishMorphology.create_with_defaults()
    morphology = _ZEMBEREK_MORPHOLOGY
    return _analyze_with_morphology(morphology, words, sentence_text, use_disambiguation)


def get_case_prediction_zemberek(analyses: List[str]) -> Optional[str]:
    """Predict UD Case from Zemberek analyses (majority vote)."""
    if not analyses:
        return None
    case_counts = defaultdict(int)
    for analysis in analyses:
        case = extract_case_from_zemberek(analysis)
        if case:
            case_counts[case] += 1
    if not case_counts:
        return None
    max_count = max(case_counts.values())
    candidates = [c for c, cnt in case_counts.items() if cnt == max_count]
    for case in CASE_PRIORITY:
        if case in candidates:
            return case
    return candidates[0] if candidates else None
