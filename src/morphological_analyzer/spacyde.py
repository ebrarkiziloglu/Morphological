"""
spaCy German (transformer or core) morphological analyser helpers.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_SPACY_MODELS: Dict[str, Any] = {}

from .trmor import CASE_PRIORITY


def extract_case_from_spacy_de(analysis: str) -> Optional[str]:
    """Extract UD Case from spaCy German morph string."""
    if not analysis:
        return None
    for part in analysis.split("|"):
        part = part.strip()
        if part.startswith("Case="):
            return part.split("=", 1)[1].strip()
    return None


def run_spacy_de_analyser(
    words: List[str],
    sentence_tokens: Optional[List[Dict[str, Any]]] = None,
    sentence_text: Optional[str] = None,
    model_name: str = "de_dep_news_trf",
    require_gpu: bool = False,
) -> Tuple[dict, set]:
    """Run spaCy German morphological analyser."""
    global _SPACY_MODELS

    try:
        import spacy
    except ImportError:
        print(
            "Error: spacy not installed. Install with: "
            "pip install spacy && python -m spacy download de_dep_news_trf"
        )
        return {}, set(words) if words else set()

    if not words and not (sentence_tokens and sentence_text):
        return {}, set()

    if model_name in _SPACY_MODELS:
        nlp = _SPACY_MODELS[model_name]
    else:
        # For transformer models (e.g. de_dep_news_trf), GPU is used via PyTorch (not CuPy).
        # We require PyTorch to see CUDA when require_gpu=True; we do not call spacy.require_gpu()
        # (that would require CuPy, which is optional for transformer models).
        if "trf" in model_name:
            try:
                spacy.prefer_gpu()  # Use GPU for thinc/CuPy if available
            except Exception:
                pass
            try:
                import torch
                torch_cuda = torch.cuda.is_available()
                print(f"spaCy transformer model: PyTorch CUDA available = {torch_cuda}")
                if require_gpu and not torch_cuda:
                    raise ValueError(
                        "GPU required but PyTorch cannot see CUDA. "
                        "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
                    )
            except ImportError:
                if require_gpu:
                    raise ValueError("GPU required but PyTorch is not installed. Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print("Error: spaCy model not found. Run: python -m spacy download " + model_name)
            return {}, set(words) if words else set()
        _SPACY_MODELS[model_name] = nlp

    # Morphologizer must be in the pipeline for token.morph; de_core_news_lg includes it
    if "morphologizer" not in nlp.pipe_names:
        import warnings
        warnings.warn(
            f"spaCy model '{model_name}' has no 'morphologizer' in pipeline "
            f"({nlp.pipe_names}). token.morph may be empty."
        )

    analyses_dict = defaultdict(list)
    no_result_words = set()

    if sentence_tokens is not None and sentence_text:
        try:
            doc = nlp(sentence_text)
            if len(doc) == len(sentence_tokens):
                for i, token in enumerate(doc):
                    form = sentence_tokens[i]["form"]
                    morph_str = str(token.morph)
                    analyses_dict[form].append(morph_str)
            else:
                # Align by form: UD and spaCy tokenization often differ
                used = set()
                for ud in sentence_tokens:
                    form = ud["form"]
                    morph_str = None
                    for j, spacy_tok in enumerate(doc):
                        if j not in used and spacy_tok.text == form:
                            morph_str = str(spacy_tok.morph)
                            used.add(j)
                            break
                    if morph_str is not None:
                        analyses_dict[form].append(morph_str)
                    else:
                        # No matching spaCy token (e.g. contraction); analyze word in context
                        try:
                            sub_doc = nlp(form)
                            if len(sub_doc) >= 1:
                                analyses_dict[form].append(str(sub_doc[0].morph))
                            else:
                                no_result_words.add(form)
                        except Exception:
                            no_result_words.add(form)
        except Exception:
            # Fallback: analyze each word independently, batched for speed.
            for w, doc in zip(words, nlp.pipe(words, batch_size=1000)):
                try:
                    if len(doc) >= 1:
                        analyses_dict[w].append(str(doc[0].morph))
                    else:
                        no_result_words.add(w)
                except Exception:
                    no_result_words.add(w)
    else:
        # No sentence context: analyze words independently in batches for speed.
        for w, doc in zip(words, nlp.pipe(words, batch_size=1000)):
            try:
                if len(doc) >= 1:
                    analyses_dict[w].append(str(doc[0].morph))
                else:
                    no_result_words.add(w)
            except Exception:
                no_result_words.add(w)

    for w in words:
        if w not in analyses_dict and w not in no_result_words:
            no_result_words.add(w)

    return dict(analyses_dict), no_result_words


def get_case_prediction_spacy_de(analyses: List[str]) -> Optional[str]:
    """Predict UD Case from spaCy German analyses (majority vote)."""
    if not analyses:
        return None
    case_counts = defaultdict(int)
    for analysis in analyses:
        case = extract_case_from_spacy_de(analysis)
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