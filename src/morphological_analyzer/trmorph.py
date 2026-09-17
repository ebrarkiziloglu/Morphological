"""
TRmorph (foma/flookup) morphological analyser helpers.
"""

import os
import subprocess
import tempfile
from collections import defaultdict
from typing import List, Optional, Tuple

from .trmor import get_final_case, get_trmor_case_prediction

TRMORPH_CASE_TAG_MAP = {
    '<acc>': '<Acc>', '<nom>': '<Nom>', '<dat>': '<Dat>', '<gen>': '<Gen>',
    '<loc>': '<Loc>', '<abl>': '<Abl>', '<ins>': '<Ins>',
}


def _parse_flookup_output(output: str, words_in_order: List[str]) -> Tuple[dict, set]:
    """Parse flookup (TRmorph) batch output."""
    analyses = defaultdict(list)
    no_result_words = set()
    lines = [line.strip() for line in output.split('\n')]
    word_index = 0
    i = 0
    while i < len(lines) and word_index < len(words_in_order):
        word = words_in_order[word_index]
        word_analyses = []
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line:
                break
            if '\t' in line:
                _, analysis = line.split('\t', 1)
                word_analyses.append(analysis.strip())
            else:
                word_analyses.append(line)
        if word_analyses:
            analyses[word] = word_analyses
        else:
            no_result_words.add(word)
        word_index += 1
    while word_index < len(words_in_order):
        no_result_words.add(words_in_order[word_index])
        word_index += 1
    return dict(analyses), no_result_words


def _normalize_trmorph_case_tags(analysis: str) -> str:
    """Convert TRmorph lowercase case tags to capital form."""
    s = analysis
    for low, cap in TRMORPH_CASE_TAG_MAP.items():
        s = s.replace(low, cap)
    return s


def _trmorph_analysis_is_noun(analysis: str) -> bool:
    """Check if TRmorph analysis indicates a noun (contains <n> or <N> tag)."""
    return '<n>' in analysis or '<N>' in analysis


def _trmorph_infer_nominative_if_noun(analyses: List[str]) -> Optional[str]:
    """
    TRmorph does not output a Nom suffix; bare nouns (no Acc, Dat, Loc, etc.) are Nominative.
    If any analysis is a noun with no case tag, return 'Nom'.
    """
    for analysis in analyses:
        if _trmorph_analysis_is_noun(analysis) and get_final_case(analysis) is None:
            return 'Nom'
    return None


def run_trmorph_analyser(
    words: List[str],
    trmorph_fst_path: str = "morphological_analyzer/TRmorph/trmorph.fst",
    flookup_path: str = "flookup",
) -> Tuple[dict, set]:
    """Run TRmorph via flookup."""
    if not words:
        return {}, set()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for w in words:
            f.write(w + '\n')
        input_file = f.name
    try:
        with open(input_file, 'r', encoding='utf-8') as inf:
            input_content = inf.read()
        cmd = [flookup_path, '-b', '-x', os.path.abspath(trmorph_fst_path)]
        result = subprocess.run(
            cmd,
            input=input_content,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300,
        )
        if result.returncode != 0:
            if result.stderr:
                print(f"flookup stderr: {result.stderr}")
            return {}, set(words)
        return _parse_flookup_output(result.stdout, words)
    except FileNotFoundError:
        print("Error: flookup not found. Install foma and ensure 'flookup' is in PATH.")
        return {}, set(words)
    except subprocess.TimeoutExpired:
        print("Error: flookup timed out.")
        return {}, set(words)
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)


def get_case_prediction_trmorph(
    analyses: List[str],
    *,
    trmorph_mode: str = 'first',
) -> Optional[str]:
    """Predict UD Case from TRmorph analyses (first analysis or majority vote)."""
    if not analyses:
        return None
    if trmorph_mode == 'first':
        analyses = [_normalize_trmorph_case_tags(analyses[0])] if analyses else []
    else:
        analyses = [_normalize_trmorph_case_tags(a) for a in analyses]
    case = get_trmor_case_prediction(analyses) if analyses else None
    if case is None and analyses:
        case = _trmorph_infer_nominative_if_noun(analyses)
    return case
