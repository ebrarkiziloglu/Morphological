"""
Helper functions for UD dataset analysis with morphological analysers.

Includes:
- Turkish lowercase conversion
- CoNLL-U parsing and dataset loading
- TRMOR/TRmorph/Zemberek case prediction
- Morphological analyzer backends
- Validation pipeline utilities
"""

import os
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

# Analyzer backends (each has run_* and get_case_prediction_*)
from morphological_analyzer import trmor, trmorph, zemberek, zeyrek, vnlp, smor, spacyde


# ============================================================
# Turkish Lowercase Conversion
# ============================================================

TURKISH_UPPER_TO_LOWER = {
    'İ': 'i',
    'I': 'ı',
    'Ö': 'ö',
    'Ü': 'ü',
    'Ğ': 'ğ',
    'Ç': 'ç',
    'Ş': 'ş',
}


def turkish_lower(text: str) -> str:
    """
    Convert text to lowercase using Turkish-specific rules.
    """
    result = []
    for char in text:
        if char in TURKISH_UPPER_TO_LOWER:
            result.append(TURKISH_UPPER_TO_LOWER[char])
        else:
            result.append(char.lower())
    return ''.join(result)


# ============================================================
# CoNLL-U Parser
# ============================================================

def parse_conllu_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse a CoNLL-U file and return a list of sentences.
    """
    sentences = []
    current_sentence = {'tokens': []}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_sentence['tokens']:
                    sentences.append(current_sentence)
                current_sentence = {'tokens': []}
                continue

            if line.startswith('#'):
                if line.startswith('# sent_id'):
                    current_sentence['sent_id'] = line.split('=', 1)[1].strip()
                elif line.startswith('# text'):
                    current_sentence['text'] = line.split('=', 1)[1].strip()
                continue

            fields = line.split('\t')
            if len(fields) != 10:
                continue

            token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields

            if '-' in token_id:
                continue

            feats_dict = {}
            if feats != '_':
                for feat in feats.split('|'):
                    if '=' in feat:
                        key, value = feat.split('=', 1)
                        feats_dict[key] = value

            token = {
                'id': token_id,
                'form': form,
                'lemma': lemma,
                'upos': upos,
                'xpos': xpos,
                'feats': feats_dict,
                'feats_str': feats,
                'head': head,
                'deprel': deprel,
                'deps': deps,
                'misc': misc
            }
            current_sentence['tokens'].append(token)

        if current_sentence['tokens']:
            sentences.append(current_sentence)

    return sentences


# ConLL-U filenames per UD treebank (for load_ud_dataset).
UD_TURKISH_BOUN_SPLITS = {
    'train': 'tr_boun-ud-train.conllu',
    'dev': 'tr_boun-ud-dev.conllu',
    'test': 'tr_boun-ud-test.conllu',
}
UD_GERMAN_GSD_SPLITS = {
    'train': 'de_gsd-ud-train.conllu',
    'dev': 'de_gsd-ud-dev.conllu',
    'test': 'de_gsd-ud-test.conllu',
}


def load_ud_dataset(
    base_path: str = "data/UD_Turkish-BOUN",
    splits_filemap: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict]]:
    """Load UD dataset splits from base_path. Uses splits_filemap for filenames if provided."""
    dataset = {}
    splits = splits_filemap if splits_filemap is not None else UD_TURKISH_BOUN_SPLITS

    for split_name, filename in splits.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            dataset[split_name] = parse_conllu_file(filepath)
            print(f"Loaded {split_name}: {len(dataset[split_name])} sentences")

    return dataset


# ============================================================
# Morphological analyzer backends (implementations in morphological_analyzer/)
# ============================================================

ANALYZER_CHOICES = ['trmor', 'trmorph', 'zemberek', 'zeyrek', 'vnlp', 'smor', 'spacy_de']


def run_analyzer(
    analyzer_name: str,
    words: List[str],
    **kwargs: Any,
) -> Tuple[Dict[str, List[str]], set]:
    """Run the selected morphological analyser on a list of words."""
    if analyzer_name == 'trmor':
        return trmor.run_trmor_analyser(
            words,
            morph_path=kwargs.get('morph_path', 'morphological_analyzer/TRMOR/morph.ca'),
            fst_infl2_path=kwargs.get('fst_infl2_path', 'morphological_analyzer/SMOR/bin/fst-infl2'),
        )
    if analyzer_name == 'trmorph':
        return trmorph.run_trmorph_analyser(
            words,
            trmorph_fst_path=kwargs.get('trmorph_fst_path', 'morphological_analyzer/TRmorph/trmorph.fst'),
            flookup_path=kwargs.get('flookup_path', 'flookup'),
        )
    if analyzer_name == 'zemberek':
        return zemberek.run_zemberek_analyser(
            words,
            sentence_text=kwargs.get('sentence_text'),
            use_disambiguation=kwargs.get('use_disambiguation', True),
            max_workers=kwargs.get('max_workers'),
        )
    if analyzer_name == 'zeyrek':
        return zeyrek.run_zeyrek_analyser(words, sentence_text=kwargs.get('sentence_text'))
    if analyzer_name == 'vnlp':
        return vnlp.run_vnlp_analyser(
            words,
            sentence_tokens=kwargs.get('sentence_tokens'),
            sentence_text=kwargs.get('sentence_text'),
            batch_size=kwargs.get('vnlp_batch_size', 128),
            require_gpu=kwargs.get('vnlp_require_gpu', False),
            memory_limit=kwargs.get('vnlp_memory_limit', None),
        )
    if analyzer_name == 'smor':
        return smor.run_smor_analyser(
            words,
            smor_infl_path=kwargs.get('smor_infl_path', 'morphological_analyzer/SMOR/smor-infl'),
        )
    if analyzer_name == 'spacy_de':
        return spacyde.run_spacy_de_analyser(
            words,
            sentence_tokens=kwargs.get('sentence_tokens'),
            sentence_text=kwargs.get('sentence_text'),
            model_name=kwargs.get('model_name', 'de_dep_news_trf'),
            require_gpu=kwargs.get('spacy_require_gpu', False),
        )
    raise ValueError(f"Unknown analyzer: {analyzer_name}. Choose from: {ANALYZER_CHOICES}")


TRMORPH_CASE_MODE_FIRST = 'first'
TRMORPH_CASE_MODE_MAJORITY = 'majority'
TRMORPH_CASE_MODES = (TRMORPH_CASE_MODE_FIRST, TRMORPH_CASE_MODE_MAJORITY)


def get_case_prediction(
    analyses: List[str],
    analyzer_name: str,
    *,
    trmorph_mode: str = TRMORPH_CASE_MODE_FIRST,
) -> Optional[str]:
    """Predict UD Case from a list of analyses using majority voting (or first-only for TRmorph when requested)."""
    if not analyses:
        return None
    if analyzer_name == 'zemberek':
        return zemberek.get_case_prediction_zemberek(analyses)
    if analyzer_name == 'zeyrek':
        return zeyrek.get_case_prediction_zeyrek(analyses)
    if analyzer_name == 'vnlp':
        return vnlp.get_case_prediction_vnlp(analyses)
    if analyzer_name == 'trmorph':
        print(f"TRMORPH MODE: {trmorph_mode}")
        return trmorph.get_case_prediction_trmorph(analyses, trmorph_mode=trmorph_mode)
    if analyzer_name == 'smor':
        return smor.get_case_prediction_smor(analyses)
    if analyzer_name == 'spacy_de':
        return spacyde.get_case_prediction_spacy_de(analyses)
    # trmor and any other TRMOR-format analyser
    return trmor.get_trmor_case_prediction(analyses)


# ============================================================
# Validation Pipeline
# ============================================================

def analyze_sentence_with_analyzer(
    sentence: Dict,
    analyzer_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyze a single UD sentence with the chosen analyser and compare with gold labels."""
    tokens = sentence['tokens']
    use_preserve_case = analyzer_name in ('smor', 'spacy_de')
    if use_preserve_case:
        words = [t['form'] for t in tokens]
    else:
        words = [turkish_lower(t['form']) for t in tokens]

    # Optional precomputed analyses (used e.g. for SMOR to batch-process all words once)
    precomputed_analyses = kwargs.pop('precomputed_analyses', None)
    precomputed_no_analysis_words = kwargs.pop('precomputed_no_analysis_words', None)

    if precomputed_analyses is not None and precomputed_no_analysis_words is not None:
        analyses = precomputed_analyses
        no_analysis_words = precomputed_no_analysis_words
    else:
        kwargs_analyze = dict(kwargs)
        if analyzer_name == 'zemberek':
            kwargs_analyze['sentence_text'] = sentence.get('text', '')
            kwargs_analyze['use_disambiguation'] = True
        if analyzer_name == 'zeyrek':
            kwargs_analyze['sentence_text'] = sentence.get('text', '')
        if analyzer_name == 'vnlp':
            kwargs_analyze['sentence_text'] = sentence.get('text', '')
            kwargs_analyze['sentence_tokens'] = tokens
        if analyzer_name == 'spacy_de':
            kwargs_analyze['sentence_text'] = sentence.get('text', '')
            kwargs_analyze['sentence_tokens'] = tokens
        analyses, no_analysis_words = run_analyzer(analyzer_name, words, **kwargs_analyze)

    lemmas_to_analyze = set()
    for token in tokens:
        if use_preserve_case:
            lemma_key = token['lemma']
            word_key = token['form']
        else:
            lemma_key = turkish_lower(token['lemma'])
            word_key = turkish_lower(token['form'])
        if lemma_key != word_key and lemma_key not in analyses:
            lemmas_to_analyze.add(lemma_key)

    lemma_analyses = {}
    if lemmas_to_analyze:
        kwargs_lemma = {k: v for k, v in kwargs.items() if k != 'sentence_text'}
        if analyzer_name == 'zemberek':
            kwargs_lemma['use_disambiguation'] = False
        if analyzer_name == 'zeyrek':
            kwargs_lemma['sentence_text'] = None
        if analyzer_name == 'vnlp':
            kwargs_lemma['sentence_text'] = None
            kwargs_lemma['sentence_tokens'] = None
        if analyzer_name == 'spacy_de':
            kwargs_lemma['sentence_text'] = None
            kwargs_lemma['sentence_tokens'] = None
        lemma_analyses, _ = run_analyzer(analyzer_name, list(lemmas_to_analyze), **kwargs_lemma)

    token_results = []
    for token in tokens:
        word = token['form']
        word_key = word if use_preserve_case else turkish_lower(word)
        lemma = token['lemma']
        lemma_key = lemma if use_preserve_case else turkish_lower(lemma)
        gold_case = token['feats'].get('Case')
        upos = token['upos']

        word_analyses = analyses.get(word_key, [])
        trmorph_mode = kwargs.get('trmorph_mode', TRMORPH_CASE_MODE_FIRST)
        predicted_case = get_case_prediction(word_analyses, analyzer_name, trmorph_mode=trmorph_mode)
        prediction_source = 'form' if predicted_case else None

        if predicted_case is None and lemma_key != word_key:
            lemma_word_analyses = analyses.get(lemma_key, []) or lemma_analyses.get(lemma_key, [])
            if lemma_word_analyses:
                predicted_case = get_case_prediction(lemma_word_analyses, analyzer_name, trmorph_mode=trmorph_mode)
                if predicted_case:
                    prediction_source = 'lemma'
                    word_analyses = lemma_word_analyses

        is_correct = (predicted_case == gold_case) if gold_case is not None else None
        num_analyses = sum(word_analyses.values()) if hasattr(word_analyses, 'values') else len(word_analyses)
        has_analysis = word_key not in no_analysis_words and (num_analyses > 0 if hasattr(word_analyses, 'values') else len(word_analyses) > 0)

        token_results.append({
            'form': word,
            'lemma': lemma,
            'upos': upos,
            'gold_case': gold_case,
            'predicted_case': predicted_case,
            'prediction_source': prediction_source,
            'is_correct': is_correct,
            'num_analyses': num_analyses,
            'has_analysis': has_analysis
        })

    return {
        'sent_id': sentence.get('sent_id', ''),
        'text': sentence.get('text', ''),
        'tokens': token_results
    }


def analyze_sentence_with_trmor(
    sentence: Dict,
    morph_path: str = "morphological_analyzer/TRMOR/morph.ca",
    fst_infl2_path: str = "morphological_analyzer/SMOR/bin/fst-infl2"
) -> Dict[str, Any]:
    """Thin wrapper: analyze a single UD sentence with TRMOR."""
    return analyze_sentence_with_analyzer(
        sentence, 'trmor', morph_path=morph_path, fst_infl2_path=fst_infl2_path
    )


def validate_on_dataset(
    sentences: List[Dict],
    analyzer_name: str,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Validate the chosen morphological analyser on a list of UD sentences."""
    # Batch precomputation: run analyser once on all unique words/lemmas to avoid
    # re-initializing per sentence. Skip for zemberek, zeyrek so they keep
    # per-sentence disambiguation (sentence_text / sentence_tokens). vnlp has its
    # own dataset-level precomputation through VNLPAnalyzer.
    precomputed_analyses = None
    precomputed_no_analysis_words = None
    # spacy_de excluded from batch: morphologizer needs sentence context for accuracy
    # vnlp now supports dataset-level precomputation via VNLPAnalyzer.
    use_batch = analyzer_name in ('trmor', 'trmorph', 'smor', 'vnlp')
    use_preserve_case = analyzer_name in ('smor', 'spacy_de')

    if use_batch:
        all_words = set()
        all_lemmas = set()
        for sentence in sentences:
            for token in sentence.get('tokens', []):
                form = token.get('form')
                lemma = token.get('lemma')
                if form:
                    all_words.add(form if use_preserve_case else turkish_lower(form))
                if lemma:
                    all_lemmas.add(lemma if use_preserve_case else turkish_lower(lemma))
        batch_words = sorted(all_words | all_lemmas)

        if batch_words:
            if analyzer_name == 'trmor':
                precomputed_analyses, precomputed_no_analysis_words = trmor.run_trmor_analyser(
                    batch_words,
                    morph_path=kwargs.get('morph_path', 'morphological_analyzer/TRMOR/morph.ca'),
                    fst_infl2_path=kwargs.get('fst_infl2_path', 'morphological_analyzer/SMOR/bin/fst-infl2'),
                )
            elif analyzer_name == 'trmorph':
                precomputed_analyses, precomputed_no_analysis_words = trmorph.run_trmorph_analyser(
                    batch_words,
                    trmorph_fst_path=kwargs.get('trmorph_fst_path', 'morphological_analyzer/TRmorph/trmorph.fst'),
                    flookup_path=kwargs.get('flookup_path', 'flookup'),
                )
            elif analyzer_name == 'smor':
                precomputed_analyses, precomputed_no_analysis_words = smor.run_smor_analyser(
                    batch_words,
                    smor_infl_path=kwargs.get('smor_infl_path', 'morphological_analyzer/SMOR/smor-infl'),
                )
            elif analyzer_name == 'vnlp':
                precomputed_analyses, precomputed_no_analysis_words = vnlp.precompute_vnlp_dataset(
                    sentences,
                    vnlp_batch_size=kwargs.get('vnlp_batch_size', 128),
                    vnlp_require_gpu=kwargs.get('vnlp_require_gpu', False),
                    memory_limit=kwargs.get('vnlp_memory_limit', None),
                )
            elif analyzer_name == 'spacy_de':
                precomputed_analyses, precomputed_no_analysis_words = spacyde.run_spacy_de_analyser(
                    batch_words,
                    sentence_tokens=None,
                    sentence_text=None,
                    model_name=kwargs.get('model_name', 'de_dep_news_trf'),
                    require_gpu=kwargs.get('spacy_require_gpu', False),
                )

    results = []
    stats = {
        'total_tokens': 0,
        'tokens_with_gold_case': 0,
        'tokens_with_prediction': 0,
        'correct_predictions': 0,
        'tokens_no_analysis': 0,
        'predictions_from_form': 0,
        'predictions_from_lemma': 0,
        'correct_from_form': 0,
        'correct_from_lemma': 0,
        'per_case': defaultdict(lambda: {'total': 0, 'correct': 0, 'predicted_as': defaultdict(int)})
    }

    for i, sentence in enumerate(sentences):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processing sentence {i + 1}/{len(sentences)}...")

        if precomputed_analyses is not None and precomputed_no_analysis_words is not None:
            result = analyze_sentence_with_analyzer(
                sentence,
                analyzer_name,
                precomputed_analyses=precomputed_analyses,
                precomputed_no_analysis_words=precomputed_no_analysis_words,
                **kwargs,
            )
        else:
            result = analyze_sentence_with_analyzer(sentence, analyzer_name, **kwargs)
        results.append(result)

        for token in result['tokens']:
            stats['total_tokens'] += 1
            if not token['has_analysis']:
                stats['tokens_no_analysis'] += 1
            gold_case = token['gold_case']
            pred_case = token['predicted_case']
            if gold_case is not None:
                stats['tokens_with_gold_case'] += 1
                stats['per_case'][gold_case]['total'] += 1
                if pred_case is not None:
                    stats['tokens_with_prediction'] += 1
                    stats['per_case'][gold_case]['predicted_as'][pred_case] += 1
                    pred_source = token.get('prediction_source')
                    if pred_source == 'form':
                        stats['predictions_from_form'] += 1
                    elif pred_source == 'lemma':
                        stats['predictions_from_lemma'] += 1
                    if token['is_correct']:
                        stats['correct_predictions'] += 1
                        stats['per_case'][gold_case]['correct'] += 1
                        if pred_source == 'form':
                            stats['correct_from_form'] += 1
                        elif pred_source == 'lemma':
                            stats['correct_from_lemma'] += 1

    if stats['tokens_with_gold_case'] > 0:
        stats['accuracy'] = stats['correct_predictions'] / stats['tokens_with_gold_case']
    else:
        stats['accuracy'] = 0.0
    if stats['tokens_with_prediction'] > 0:
        stats['accuracy_when_predicted'] = stats['correct_predictions'] / stats['tokens_with_prediction']
    else:
        stats['accuracy_when_predicted'] = 0.0

    stats['per_case'] = {k: dict(v) for k, v in stats['per_case'].items()}
    for case in stats['per_case']:
        stats['per_case'][case]['predicted_as'] = dict(stats['per_case'][case]['predicted_as'])

    # Precision and recall for key cases (Accusative, Dative)
    target_cases = ['Acc', 'Dat']
    precision_recall = {}
    for case in target_cases:
        if case not in stats['per_case']:
            continue
        case_stats = stats['per_case'][case]
        tp = case_stats['correct']
        total_gold = case_stats['total']
        total_pred_as_case = 0
        for gold_case, gold_stats in stats['per_case'].items():
            total_pred_as_case += gold_stats['predicted_as'].get(case, 0)
        precision = (tp / total_pred_as_case) if total_pred_as_case > 0 else 0.0
        recall = (tp / total_gold) if total_gold > 0 else 0.0
        precision_recall[case] = {'precision': precision, 'recall': recall}
    stats['precision_recall'] = precision_recall

    return results, stats


def validate_trmor_on_dataset(
    sentences: List[Dict],
    morph_path: str = "morphological_analyzer/TRMOR/morph.ca",
    fst_infl2_path: str = "morphological_analyzer/SMOR/bin/fst-infl2",
    verbose: bool = False
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Validate TRMOR on a list of UD sentences."""
    return validate_on_dataset(
        sentences, 'trmor', verbose=verbose,
        morph_path=morph_path, fst_infl2_path=fst_infl2_path
    )


def print_statistics(stats: Dict[str, Any], split_name: str = "", analyzer_label: str = "TRMOR"):
    """Print validation statistics."""
    print("\n" + "=" * 70)
    print(f"{analyzer_label.upper()} VALIDATION STATISTICS {('(' + split_name + ')') if split_name else ''}")
    print("=" * 70)

    print(f"\n📊 Overall Statistics:")
    print(f"   Total tokens:                    {stats['total_tokens']:,}")
    print(f"   Tokens with gold Case tag:       {stats['tokens_with_gold_case']:,}")
    print(f"   Tokens with {analyzer_label} prediction:    {stats['tokens_with_prediction']:,}")
    print(f"   Correct predictions:             {stats['correct_predictions']:,}")
    print(f"   Tokens with no {analyzer_label} analysis:   {stats['tokens_no_analysis']:,}")

    print(f"\n📈 Accuracy:")
    print(f"   Overall accuracy (gold has Case): {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
    print(f"   Accuracy when {analyzer_label} predicts:     {stats['accuracy_when_predicted']:.4f} ({stats['accuracy_when_predicted']*100:.2f}%)")

    precision_recall = stats.get('precision_recall', {})
    if precision_recall:
        print(f"\n🎯 Precision / Recall (Acc, Dat):")
        for case in ['Acc', 'Dat']:
            if case in precision_recall:
                prec = precision_recall[case]['precision']
                rec = precision_recall[case]['recall']
                print(
                    f"   {case:>3} - Precision: {prec*100:.2f} | "
                    f"Recall: {rec*100:.2f}"
                )

    pred_form = stats.get('predictions_from_form', 0)
    pred_lemma = stats.get('predictions_from_lemma', 0)
    correct_form = stats.get('correct_from_form', 0)
    correct_lemma = stats.get('correct_from_lemma', 0)

    print(f"\n📊 Prediction Source Breakdown:")
    print(f"   Predictions from word form:      {pred_form:,}")
    if pred_form > 0:
        print(f"     - Correct:                      {correct_form:,} ({correct_form/pred_form*100:.2f}%)")
    print(f"   Predictions from lemma fallback: {pred_lemma:,}")
    if pred_lemma > 0:
        print(f"     - Correct:                      {correct_lemma:,} ({correct_lemma/pred_lemma*100:.2f}%)")

    print(f"\n📊 Per-Case Performance:")
    print(f"{'Case':<8} | {'Total':>8} | {'Correct':>8} | {'Accuracy':>10} | {'Top Confusion':>20}")
    print("-" * 70)

    for case in ['Nom', 'Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins']:
        if case in stats['per_case']:
            case_stats = stats['per_case'][case]
            total = case_stats['total']
            correct = case_stats['correct']
            accuracy = correct / total if total > 0 else 0

            predicted_as = case_stats.get('predicted_as', {})
            wrong_preds = {k: v for k, v in predicted_as.items() if k != case}
            if wrong_preds:
                top_confusion = max(wrong_preds.items(), key=lambda x: x[1])
                confusion_str = f"{top_confusion[0]}:{top_confusion[1]}"
            else:
                confusion_str = "-"

            print(f"{case:<8} | {total:>8,} | {correct:>8,} | {accuracy:>9.2%} | {confusion_str:>20}")

    print("=" * 70)


def save_results(
    results: List[Dict],
    stats: Dict[str, Any],
    output_dir: str,
    split_name: str
):
    """Save validation results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f"{split_name}_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    stats_path = os.path.join(output_dir, f"{split_name}_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved:")
    print(f"   - {results_path}")
    print(f"   - {stats_path}")


def combine_statistics(all_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Combine statistics from multiple splits into one aggregate."""
    combined = {
        'total_tokens': 0,
        'tokens_with_gold_case': 0,
        'tokens_with_prediction': 0,
        'correct_predictions': 0,
        'tokens_no_analysis': 0,
        'predictions_from_form': 0,
        'predictions_from_lemma': 0,
        'correct_from_form': 0,
        'correct_from_lemma': 0,
        'per_case': defaultdict(lambda: {'total': 0, 'correct': 0, 'predicted_as': defaultdict(int)})
    }

    for split_name, stats in all_stats.items():
        combined['total_tokens'] += stats.get('total_tokens', 0)
        combined['tokens_with_gold_case'] += stats.get('tokens_with_gold_case', 0)
        combined['tokens_with_prediction'] += stats.get('tokens_with_prediction', 0)
        combined['correct_predictions'] += stats.get('correct_predictions', 0)
        combined['tokens_no_analysis'] += stats.get('tokens_no_analysis', 0)
        combined['predictions_from_form'] += stats.get('predictions_from_form', 0)
        combined['predictions_from_lemma'] += stats.get('predictions_from_lemma', 0)
        combined['correct_from_form'] += stats.get('correct_from_form', 0)
        combined['correct_from_lemma'] += stats.get('correct_from_lemma', 0)

        for case, case_stats in stats.get('per_case', {}).items():
            combined['per_case'][case]['total'] += case_stats.get('total', 0)
            combined['per_case'][case]['correct'] += case_stats.get('correct', 0)
            for pred_case, count in case_stats.get('predicted_as', {}).items():
                combined['per_case'][case]['predicted_as'][pred_case] += count

    if combined['tokens_with_gold_case'] > 0:
        combined['accuracy'] = combined['correct_predictions'] / combined['tokens_with_gold_case']
    else:
        combined['accuracy'] = 0.0

    if combined['tokens_with_prediction'] > 0:
        combined['accuracy_when_predicted'] = combined['correct_predictions'] / combined['tokens_with_prediction']
    else:
        combined['accuracy_when_predicted'] = 0.0

    combined['per_case'] = {k: dict(v) for k, v in combined['per_case'].items()}
    for case in combined['per_case']:
        combined['per_case'][case]['predicted_as'] = dict(combined['per_case'][case]['predicted_as'])

    # Precision and recall for key cases (Accusative, Dative) on combined stats
    target_cases = ['Acc', 'Dat']
    precision_recall = {}
    for case in target_cases:
        if case not in combined['per_case']:
            continue
        case_stats = combined['per_case'][case]
        tp = case_stats['correct']
        total_gold = case_stats['total']
        total_pred_as_case = 0
        for gold_case, gold_stats in combined['per_case'].items():
            total_pred_as_case += gold_stats['predicted_as'].get(case, 0)
        precision = (tp / total_pred_as_case) if total_pred_as_case > 0 else 0.0
        recall = (tp / total_gold) if total_gold > 0 else 0.0
        precision_recall[case] = {'precision': precision, 'recall': recall}
    combined['precision_recall'] = precision_recall

    return combined


def print_combined_statistics(all_stats: Dict[str, Dict[str, Any]], output_dir: str = None):
    """Print combined statistics table comparing all splits side by side, plus aggregate."""
    splits = list(all_stats.keys())
    combined = combine_statistics(all_stats)

    print("\n")
    print("=" * 90)
    print("COMBINED VALIDATION STATISTICS (ALL SPLITS)")
    print("=" * 90)

    header = f"{'Metric':<35}"
    for split in splits:
        header += f" | {split:>12}"
    header += f" | {'TOTAL':>12}"
    print(header)
    print("-" * 90)

    metrics = [
        ('Total tokens', 'total_tokens'),
        ('Tokens with gold Case', 'tokens_with_gold_case'),
        ('Tokens with TRMOR prediction', 'tokens_with_prediction'),
        ('Correct predictions', 'correct_predictions'),
        ('Tokens with no analysis', 'tokens_no_analysis'),
        ('Predictions from form', 'predictions_from_form'),
        ('Predictions from lemma', 'predictions_from_lemma'),
    ]

    for label, key in metrics:
        row = f"{label:<35}"
        for split in splits:
            val = all_stats[split].get(key, 0)
            row += f" | {val:>12,}"
        row += f" | {combined.get(key, 0):>12,}"
        print(row)

    print("-" * 90)

    row = f"{'Overall accuracy':<35}"
    for split in splits:
        val = all_stats[split].get('accuracy', 0) * 100
        row += f" | {val:>11.2f}%"
    row += f" | {combined.get('accuracy', 0) * 100:>11.2f}%"
    print(row)

    row = f"{'Accuracy when predicted':<35}"
    for split in splits:
        val = all_stats[split].get('accuracy_when_predicted', 0) * 100
        row += f" | {val:>11.2f}%"
    row += f" | {combined.get('accuracy_when_predicted', 0) * 100:>11.2f}%"
    print(row)

    row = f"{'Form prediction accuracy':<35}"
    for split in splits:
        pf = all_stats[split].get('predictions_from_form', 0)
        cf = all_stats[split].get('correct_from_form', 0)
        val = (cf / pf * 100) if pf > 0 else 0
        row += f" | {val:>11.2f}%"
    pf_total = combined.get('predictions_from_form', 0)
    cf_total = combined.get('correct_from_form', 0)
    val_total = (cf_total / pf_total * 100) if pf_total > 0 else 0
    row += f" | {val_total:>11.2f}%"
    print(row)

    row = f"{'Lemma fallback accuracy':<35}"
    for split in splits:
        pl = all_stats[split].get('predictions_from_lemma', 0)
        cl = all_stats[split].get('correct_from_lemma', 0)
        val = (cl / pl * 100) if pl > 0 else 0
        row += f" | {val:>11.2f}%"
    pl_total = combined.get('predictions_from_lemma', 0)
    cl_total = combined.get('correct_from_lemma', 0)
    val_total = (cl_total / pl_total * 100) if pl_total > 0 else 0
    row += f" | {val_total:>11.2f}%"
    print(row)

    print("=" * 90)

    print("\n📊 Per-Case Performance (Combined across all splits):")
    print(f"{'Case':<8} | {'Total':>10} | {'Correct':>10} | {'Accuracy':>10} | {'Top Confusion':>25}")
    print("-" * 75)

    for case in ['Nom', 'Acc', 'Dat', 'Gen', 'Loc', 'Abl', 'Ins']:
        if case in combined['per_case']:
            case_stats = combined['per_case'][case]
            total = case_stats['total']
            correct = case_stats['correct']
            accuracy = correct / total if total > 0 else 0

            predicted_as = case_stats.get('predicted_as', {})
            wrong_preds = {k: v for k, v in predicted_as.items() if k != case}
            if wrong_preds:
                top_confusion = max(wrong_preds.items(), key=lambda x: x[1])
                confusion_str = f"{top_confusion[0]}: {top_confusion[1]:,}"
            else:
                confusion_str = "-"

            print(f"{case:<8} | {total:>10,} | {correct:>10,} | {accuracy:>9.2%} | {confusion_str:>25}")

    print("=" * 75)

    precision_recall = combined.get('precision_recall', {})
    if precision_recall:
        print("\n🎯 Combined Precision / Recall (Acc, Dat):")
        for case in ['Acc', 'Dat']:
            if case in precision_recall:
                prec = precision_recall[case]['precision']
                rec = precision_recall[case]['recall']
                print(
                    f"   {case:>3} - Precision: {prec*100:.2f} | "
                    f"Recall: {rec*100:.2f}"
                )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined_path = os.path.join(output_dir, "combined_stats.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Combined statistics saved to: {combined_path}")

    return combined


def analyzer_display_name(analyzer_name: str) -> str:
    """Return a display label for the analyzer."""
    return {
        'trmor': 'TRMOR', 'trmorph': 'TRmorph', 'zemberek': 'Zemberek', 'zeyrek': 'Zeyrek',
        'smor': 'SMOR', 'vnlp': 'vnlp', 'spacy_de': 'spaCy (de_dep_news_trf)'
    }.get(analyzer_name.lower(), analyzer_name)
