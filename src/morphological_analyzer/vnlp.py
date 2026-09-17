"""
vnlp StemmerAnalyzer morphological analyser helpers (class-based).

VNLP uses TensorFlow under the hood (GRU-based RNNs). To run on GPU and speed up
inference:
- Install TensorFlow in the version range required by VNLP (see vnlp setup.py):
  - Python >= 3.8: tensorflow>=2.6,<2.16
- Ensure CUDA/cuDNN are available so TensorFlow can see the GPU (e.g. on Slurm
  use --gres gpu:1 and load a module or conda env that provides CUDA).
- GPU is configured automatically when the analyzer is created (memory growth
  enabled). Use get_vnlp_gpu_info() to verify GPU is visible.
"""

from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

VNLP_UD_CASES = ('Nom', 'Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins')

from .trmor import CASE_PRIORITY

# Set by _configure_tensorflow_gpu() when the analyzer is first created.
_VNLP_GPU_AVAILABLE: Optional[bool] = None
_VNLP_GPU_DEVICES: List[str] = []


def _configure_tensorflow_gpu(memory_limit: Optional[int] = None) -> Tuple[bool, List[str]]:
    """
    Configure TensorFlow to use GPU with memory growth (must run before
    importing vnlp, so that VNLP's TensorFlow models use GPU).
    Returns (gpu_available, list of GPU device names).
    """
    global _VNLP_GPU_AVAILABLE, _VNLP_GPU_DEVICES
    
    import os
    if memory_limit is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    
    try:
        import tensorflow as tf
    except ImportError:
        _VNLP_GPU_AVAILABLE = False
        _VNLP_GPU_DEVICES = []
        return False, []

    gpus = tf.config.list_physical_devices("GPU")
    if memory_limit is not None:
        for gpu in gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            except RuntimeError:
                pass  # We catch this, but vnlp import will likely have already blocked us if not careful.

        # The vnlp module automatically runs `tf.config.experimental.set_memory_growth(gpu, True)` on import.
        # This crashes TensorFlow because you cannot set memory growth ONCE virtual devices are configured.
        # We monkeypatch the method to do nothing so vnlp's init script doesn't crash our script.
        tf.config.experimental.set_memory_growth = lambda device, enable: None

    _VNLP_GPU_AVAILABLE = len(gpus) > 0
    _VNLP_GPU_DEVICES = [gpu.name for gpu in gpus]
    return _VNLP_GPU_AVAILABLE, _VNLP_GPU_DEVICES


def get_vnlp_gpu_info(memory_limit: Optional[int] = None) -> Tuple[bool, str]:
    """
    Return whether VNLP is using (or will use) GPU and a short message.
    Call after get_analyzer() or any use of VNLP so that TensorFlow has been
    configured.
    """
    if _VNLP_GPU_AVAILABLE is None:
        # Not yet initialized: configure now so we can report.
        _configure_tensorflow_gpu(memory_limit=memory_limit)
    if _VNLP_GPU_AVAILABLE and _VNLP_GPU_DEVICES:
        n = len(_VNLP_GPU_DEVICES)
        devices = ", ".join(_VNLP_GPU_DEVICES)
        return True, f"VNLP TensorFlow: {n} GPU(s) available ({devices})"
    return False, "VNLP TensorFlow: no GPU visible (using CPU). Install CUDA/cuDNN and ensure TensorFlow sees the GPU."


class VNLPAnalyzer:
    """
    Class-based wrapper around vnlp.StemmerAnalyzer with:
    - single stemmer instance
    - cached sentence predictions
    - dataset-level precomputation
    - optional word-level analysis for lemmas.
    """

    _SENTENCE_CACHE_MAX = 20_000  # cap to avoid unbounded host RAM

    def __init__(self, batch_size: int = 256, memory_limit: Optional[int] = None) -> None:
        self.batch_size = batch_size
        self.memory_limit = memory_limit
        self._stemmer = self._init_stemmer()
        self._sentence_cache: Dict[str, Tuple[str, ...]] = {}
        self._sentence_cache_order: List[str] = []
        self._word_analyses: Dict[str, Counter[str]] = {}
        self._no_result_words: set = set()

    def _init_stemmer(self) -> Optional[Any]:
        # Configure TensorFlow for GPU before VNLP (and thus TF) loads models.
        _configure_tensorflow_gpu(memory_limit=self.memory_limit)
        try:
            from vnlp import StemmerAnalyzer
        except ImportError:
            print("Error: vnlp not installed. Install with: pip install vngrs-nlp")
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
        while len(self._sentence_cache) >= self._SENTENCE_CACHE_MAX and self._sentence_cache_order:
            old = self._sentence_cache_order.pop(0)
            del self._sentence_cache[old]
        self._sentence_cache[sentence] = pred
        self._sentence_cache_order.append(sentence)
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
            if key not in self._word_analyses:
                self._word_analyses[key] = Counter()
            self._word_analyses[key][analysis] += 1

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
                c = self._word_analyses[key]
                analyses_dict[key] = list(chain.from_iterable([a] * cnt for a, cnt in c.items()))
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
                    if key not in self._word_analyses:
                        self._word_analyses[key] = Counter()
                    self._word_analyses[key][analysis] += 1
            else:
                for w in remaining_words:
                    key = turkish_lower(w)
                    no_result_words.add(key)
                    self._no_result_words.add(key)

        return dict(analyses_dict), no_result_words

    def precompute_dataset(
        self,
        sentences: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Counter[str]], set]:
        """
        Precompute analyses for an entire UD dataset (sentence-by-sentence).
        Returns word -> Counter(analysis -> count) to keep host RAM bounded.
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
                if key not in self._word_analyses:
                    self._word_analyses[key] = Counter()
                self._word_analyses[key][analysis] += 1

        return dict(self._word_analyses), set(self._no_result_words)


_ANALYZER_INSTANCE: Optional[VNLPAnalyzer] = None


def get_analyzer(
    batch_size: int = 256,
    require_gpu: bool = False,
    memory_limit: Optional[int] = None,
) -> Optional[VNLPAnalyzer]:
    """Get a singleton VNLPAnalyzer instance with the requested batch_size.
    If require_gpu is True, raises ValueError when TensorFlow cannot see a GPU.
    """
    global _ANALYZER_INSTANCE
    if _ANALYZER_INSTANCE is None or _ANALYZER_INSTANCE.batch_size != batch_size or _ANALYZER_INSTANCE.memory_limit != memory_limit:
        _ANALYZER_INSTANCE = VNLPAnalyzer(batch_size=batch_size, memory_limit=memory_limit)
    if not _ANALYZER_INSTANCE.stemmer_available:
        return None
    if require_gpu:
        gpu_ok, msg = get_vnlp_gpu_info(memory_limit=memory_limit)
        if not gpu_ok:
            raise ValueError(
                "VNLP GPU required (--vnlp-gpu) but TensorFlow cannot see a GPU. "
                "Ensure CUDA and cuDNN are installed and Slurm has allocated a GPU (e.g. --gres gpu:1). "
                "See https://www.tensorflow.org/install/gpu"
            )
    return _ANALYZER_INSTANCE


def run_vnlp_analyser(
    words: List[str],
    sentence_tokens: Optional[List[Dict[str, Any]]] = None,
    sentence_text: Optional[str] = None,
    batch_size: int = 256,
    require_gpu: bool = False,
    memory_limit: Optional[int] = None,
) -> Tuple[Dict[str, List[str]], set]:
    """
    Backwards-compatible entry point used by the helpers.

    If sentence_text and sentence_tokens are given, analyze that sentence.
    Otherwise, analyze the given standalone words (e.g. lemmas).
    """
    from morphological_analysis_helpers import turkish_lower

    analyzer = get_analyzer(batch_size=batch_size, require_gpu=require_gpu, memory_limit=memory_limit)

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
    vnlp_require_gpu: bool = False,
    memory_limit: Optional[int] = None,
) -> Tuple[Dict[str, List[str]], set]:
    """
    Precompute VNLP analyses for a full UD dataset (sentence list).
    Intended to be called once per split before running evaluation.
    """
    analyzer = get_analyzer(batch_size=vnlp_batch_size, require_gpu=vnlp_require_gpu, memory_limit=memory_limit)
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


def get_case_prediction_vnlp(analyses: Union[List[str], Counter[str]]) -> Optional[str]:
    """Predict UD Case from vnlp analyses (majority vote). Accepts list or Counter(analysis -> count)."""
    if not analyses:
        return None
    case_counts = defaultdict(int)
    if hasattr(analyses, "items"):
        for analysis, cnt in analyses.items():
            case = extract_case_from_vnlp(analysis)
            if case:
                case_counts[case] += cnt
    else:
        for analysis in analyses:
            case = extract_case_from_vnlp(analysis)
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
