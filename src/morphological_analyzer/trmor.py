"""
TRMOR morphological analyser helpers.

- TRMOR-style case tags and case prediction (shared with TRmorph/SMOR)
- run_trmor_analyser (delegates to morphological_analyzer.run_morphological_analyser)
"""

from collections import defaultdict
from typing import List, Optional, Tuple

# Used by TRmorph and SMOR for case prediction
TRMOR_TO_UD_CASE = {
    '<Acc>': 'Acc',
    '<Nom>': 'Nom',
    '<Dat>': 'Dat',
    '<Gen>': 'Gen',
    '<Loc>': 'Loc',
    '<Abl>': 'Abl',
    '<Ins>': 'Ins',
}

CASE_PRIORITY = ['Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins', 'Nom']


def get_final_case(analysis: str) -> Optional[str]:
    """Get the last occurring (final) case tag from a single analysis string."""
    last_position = -1
    last_case = None

    for case in CASE_PRIORITY:
        trmor_tag = f'<{case}>'
        pos = analysis.rfind(trmor_tag)
        if pos > last_position:
            last_position = pos
            last_case = case

    return last_case


def get_trmor_case_prediction(analyses: List[str]) -> Optional[str]:
    """Predict Case using majority voting over TRMOR analyses."""
    if not analyses:
        return None

    case_counts = defaultdict(int)

    for analysis in analyses:
        case = get_final_case(analysis)
        if case:
            case_counts[case] += 1

    if not case_counts:
        return None

    max_count = max(case_counts.values())
    candidates = [case for case, count in case_counts.items() if count == max_count]

    for case in CASE_PRIORITY:
        if case in candidates:
            return case

    return candidates[0] if candidates else None


def run_trmor_analyser(
    words: List[str],
    morph_path: str = "morphological_analyzer/TRMOR/morph.ca",
    fst_infl2_path: str = "morphological_analyzer/SMOR/bin/fst-infl2",
) -> Tuple[dict, set]:
    """Run TRMOR morphological analyser on a list of words."""
    from morphological_analyzer import run_morphological_analyser
    return run_morphological_analyser(words, morph_path=morph_path, fst_infl2_path=fst_infl2_path)
