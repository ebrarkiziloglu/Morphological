"""
SMOR (German) morphological analyser helpers.
"""

import os
import subprocess
from typing import List, Optional, Tuple

from .trmor import get_trmor_case_prediction


def run_smor_analyser(
    words: List[str],
    smor_infl_path: str = "morphological_analyzer/SMOR/smor-infl",
) -> Tuple[dict, set]:
    """Run SMOR (German) via smor-infl script. Reads words from stdin, one per line.
    Output format: '> word' then one or more analysis lines (e.g. Haus<NN>Block<+NN><Masc><Nom><Pl>)."""
    if not words:
        return {}, set()

    smor_dir = os.path.dirname(os.path.abspath(smor_infl_path))
    script_name = os.path.basename(smor_infl_path)
    stdin_text = "\n".join(words) + "\n"

    try:
        proc = subprocess.run(
            [os.path.join(smor_dir, script_name) if os.path.dirname(smor_infl_path) else smor_infl_path],
            input=stdin_text.encode("utf-8"),
            capture_output=True,
            cwd=smor_dir,
            timeout=300,
            env=os.environ,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        no_result = set(words)
        if hasattr(e, "filename"):
            print(f"SMOR error ({e}): {getattr(e, 'filename', '')}")
        return {}, no_result

    analyses_dict = {}
    no_result_words = set()
    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("> "):
            current_word = line[2:].strip()
            current_analyses = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() == "" or next_line.startswith("> "):
                    break
                current_analyses.append(next_line.strip())
                i += 1
            if current_analyses:
                analyses_dict[current_word] = current_analyses
            else:
                no_result_words.add(current_word)

    for w in words:
        if w not in analyses_dict and w not in no_result_words:
            no_result_words.add(w)

    return analyses_dict, no_result_words


def get_case_prediction_smor(analyses: List[str]) -> Optional[str]:
    """Predict UD Case from SMOR analyses (TRMOR-style tags)."""
    return get_trmor_case_prediction(analyses)
