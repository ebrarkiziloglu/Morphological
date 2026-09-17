"""
vnlp StemmerAnalyzer morphological analyser helpers (class-based).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

VNLP_UD_CASES = ('Nom', 'Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins')
CASE_PRIORITY = ['Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins', 'Nom']

# from .trmor import CASE_PRIORITY

class VNLPAnalyzer:
    """
    Class-based wrapper around vnlp.StemmerAnalyzer with:
    - single stemmer instance
    - cached sentence predictions
    - dataset-level precomputation
    - optional word-level analysis for lemmas.
    """

    def __init__(self, batch_size: int = 256) -> None:
        self.batch_size = batch_size
        self._stemmer = self._init_stemmer()
        self._sentence_cache: Dict[str, Tuple[str, ...]] = {}
        self._word_analyses: Dict[str, List[str]] = defaultdict(list)
        self._no_result_words: set = set()

    def _init_stemmer(self) -> Optional[Any]:
        try:
            from vnlp import StemmerAnalyzer
        except ImportError:
            print("Error: vnlp not installed. Install with: pip install vnlp")
            return None
        try:
            return StemmerAnalyzer()
        except Exception as e:
            print(
                "Error: failed to initialize vnlp StemmerAnalyzer. "
                "This can happen on clusters without access to VNLP model weights. "
                f"Details: {e}"
            )
            return None

    @property
    def stemmer_available(self) -> bool:
        return self._stemmer is not None

    def _predict_sentence(self, sentence: str) -> Tuple[str, ...]:
        """
        Cached predict for a sentence (or any space-separated token sequence).
        Returns an empty tuple if stemmer is unavailable.
        """
        if not self.stemmer_available:
            return ()
        if sentence in self._sentence_cache:
            return self._sentence_cache[sentence]
        try:
            pred = tuple(self._stemmer.predict(sentence, batch_size=self.batch_size))
        except Exception as e:
            print(
                "Error: vnlp StemmerAnalyzer.predict failed. "
                "This often happens on clusters without internet access for downloading "
                "VNLP model weights (Stemmer_Shen_prod.weights). "
                f"Details: {e}"
            )
            return ()
        self._sentence_cache[sentence] = pred
        return pred

    def analyze_sentence_tokens(
        self,
        sentence_tokens: List[Dict[str, Any]],
        sentence_text: str,
    ) -> Tuple[Dict[str, List[str]], set]:
        """
        Analyze a single UD sentence using sentence_text + tokens.
        Updates internal caches and returns per-word analyses for that sentence.
        """
        from morphological_analysis_helpers import turkish_lower

        if not sentence_tokens or not sentence_text:
            return {}, set()

        analyses_dict: Dict[str, List[str]] = defaultdict(list)
        no_result_words: set = set()

        pred = self._predict_sentence(sentence_text)
        if not pred or len(pred) != len(sentence_tokens):
            # On mismatch, mark all words in this sentence as no-result.
            for tok in sentence_tokens:
                key = turkish_lower(tok['form'])
                no_result_words.add(key)
                self._no_result_words.add(key)
            return {}, no_result_words

        for i, tok in enumerate(sentence_tokens):
            key = turkish_lower(tok['form'])
            analysis = pred[i]
            analyses_dict[key].append(analysis)
            self._word_analyses[key].append(analysis)

        return dict(analyses_dict), no_result_words

    def analyze_words(self, words: List[str]) -> Tuple[Dict[str, List[str]], set]:
        """
        Analyze a list of standalone words (used mainly for lemma backoff).
        Uses a single space-joined pseudo sentence and caches predictions.
        """
        from morphological_analysis_helpers import turkish_lower

        if not words:
            return {}, set()

        analyses_dict: Dict[str, List[str]] = defaultdict(list)
        no_result_words: set = set()

        # Reuse any existing word-level analyses from dataset precomputation.
        remaining_words: List[str] = []
        for w in words:
            key = turkish_lower(w)
            if key in self._word_analyses:
                analyses_dict[key].extend(self._word_analyses[key])
            elif key in self._no_result_words:
                no_result_words.add(key)
            else:
                remaining_words.append(w)

        if remaining_words and self.stemmer_available:
            pseudo_sentence = " ".join(remaining_words)
            pred = self._predict_sentence(pseudo_sentence)
            if pred and len(pred) == len(remaining_words):
                for i, w in enumerate(remaining_words):
                    key = turkish_lower(w)
                    analysis = pred[i]
                    analyses_dict[key].append(analysis)
                    self._word_analyses[key].append(analysis)
            else:
                for w in remaining_words:
                    key = turkish_lower(w)
                    no_result_words.add(key)
                    self._no_result_words.add(key)

        return dict(analyses_dict), no_result_words

    def precompute_dataset(
        self,
        sentences: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, List[str]], set]:
        """
        Precompute analyses for an entire UD dataset (sentence-by-sentence).
        Populates internal word-level caches and returns a global mapping
        (word -> list of analyses) plus the set of words with no analysis.
        """
        from morphological_analysis_helpers import turkish_lower

        for sent in sentences:
            tokens = sent.get('tokens') or []
            text = sent.get('text') or ""
            if not tokens or not text:
                continue

            pred = self._predict_sentence(text)
            if not pred or len(pred) != len(tokens):
                for tok in tokens:
                    key = turkish_lower(tok['form'])
                    self._no_result_words.add(key)
                continue

            for i, tok in enumerate(tokens):
                key = turkish_lower(tok['form'])
                analysis = pred[i]
                self._word_analyses[key].append(analysis)

        return dict(self._word_analyses), set(self._no_result_words)


_ANALYZER_INSTANCE: Optional[VNLPAnalyzer] = None


def get_analyzer(batch_size: int = 256) -> Optional[VNLPAnalyzer]:
    """Get a singleton VNLPAnalyzer instance with the requested batch_size."""
    global _ANALYZER_INSTANCE
    if _ANALYZER_INSTANCE is None or _ANALYZER_INSTANCE.batch_size != batch_size:
        _ANALYZER_INSTANCE = VNLPAnalyzer(batch_size=batch_size)
    if not _ANALYZER_INSTANCE.stemmer_available:
        return None
    return _ANALYZER_INSTANCE


def run_vnlp_analyser(
    words: List[str],
    sentence_tokens: Optional[List[Dict[str, Any]]] = None,
    sentence_text: Optional[str] = None,
    batch_size: int = 256,
) -> Tuple[Dict[str, List[str]], set]:
    """
    Backwards-compatible entry point used by the helpers.

    If sentence_text and sentence_tokens are given, analyze that sentence.
    Otherwise, analyze the given standalone words (e.g. lemmas).
    """
    from morphological_analysis_helpers import turkish_lower

    analyzer = get_analyzer(batch_size=batch_size)

    if analyzer is None:
        words_set = set(turkish_lower(w) for w in words) if words else set()
        return {}, words_set

    if sentence_text and sentence_tokens:
        return analyzer.analyze_sentence_tokens(sentence_tokens, sentence_text)

    if words:
        return analyzer.analyze_words(words)

    return {}, set()


def precompute_vnlp_dataset(
    sentences: List[Dict[str, Any]],
    vnlp_batch_size: int = 256,
) -> Tuple[Dict[str, List[str]], set]:
    """
    Precompute VNLP analyses for a full UD dataset (sentence list).
    Intended to be called once per split before running evaluation.
    """
    analyzer = get_analyzer(batch_size=vnlp_batch_size)
    if analyzer is None:
        # No stemmer: report all lowercased forms in the dataset as having no analysis.
        from morphological_analysis_helpers import turkish_lower

        all_words = set()
        for sent in sentences:
            for tok in sent.get('tokens', []):
                form = tok.get('form')
                if form:
                    all_words.add(turkish_lower(form))
        return {}, all_words
    return analyzer.precompute_dataset(sentences)


def extract_case_from_vnlp(analysis: str) -> Optional[str]:
    """Extract UD Case from vnlp StemmerAnalyzer output (e.g. 'üniversite+Noun+A3sg+Pnon+Nom')."""
    if not analysis or '+' not in analysis:
        return None
    parts = analysis.split('+')
    if parts and parts[-1] in VNLP_UD_CASES:
        return parts[-1]
    return None


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
