"""
UD Dataset Analysis with Morphological Analysers

This script:
1. Loads the UD Turkish-BOUN dataset (CoNLL-U format)
2. Extracts gold standard Case tags for each word
3. Runs a chosen morphological analyser (TRMOR, TRmorph, or others) on each word
4. Compares predictions with gold labels using majority voting
5. Calculates and reports accuracy metrics
"""

import argparse
import os
import time
from typing import List

from morphological_analysis_helpers import (
    ANALYZER_CHOICES,
    TRMORPH_CASE_MODES,
    UD_GERMAN_GSD_SPLITS,
    load_ud_dataset,
    validate_on_dataset,
    print_statistics,
    save_results,
    print_combined_statistics,
    analyzer_display_name,
)


def main(
    ud_path: str = "data/UD_Turkish-BOUN",
    splits: List[str] = None,
    max_sentences: int = None,
    offset: int = 0,
    analyzer: str = "trmor",
    morph_path: str = "morphological_analyzer/TRMOR/morph.ca",
    fst_infl2_path: str = "morphological_analyzer/SMOR/bin/fst-infl2",
    trmorph_fst_path: str = "morphological_analyzer/TRmorph/trmorph.fst",
    flookup_path: str = "flookup",
    trmorph_mode: str = "first",
    smor_infl_path: str = "morphological_analyzer/SMOR/smor-infl",
    vnlp_batch_size: int = 128,
    vnlp_require_gpu: bool = False,
    vnlp_memory_limit: int = None,
    spacy_model: str = "de_dep_news_trf",
    spacy_require_gpu: bool = False,
    output_dir: str = "ud_validation_results",
    save_output: bool = True,
    verbose: bool = True
):
    """
    Main validation pipeline.
    """
    if analyzer not in ANALYZER_CHOICES:
        raise ValueError(f"Unknown analyzer '{analyzer}'. Choose from: {ANALYZER_CHOICES}")
    if analyzer == 'trmorph' and trmorph_mode not in TRMORPH_CASE_MODES:
        raise ValueError(f"trmorph_mode must be one of {TRMORPH_CASE_MODES}, got '{trmorph_mode}'")

    if analyzer in ("smor", "spacy_de"):
        ud_path = ud_path or "data/UD_German-GSD"
        dataset_label = "UD German-GSD"
    else:
        dataset_label = "UD Turkish-BOUN"

    analyzer_label = analyzer_display_name(analyzer)
    print(f"🚀 Starting {analyzer_label} validation on {dataset_label} dataset...")

    print("\n📁 Loading UD dataset...")
    if analyzer in ("smor", "spacy_de"):
        dataset = load_ud_dataset(ud_path, splits_filemap=UD_GERMAN_GSD_SPLITS)
    else:
        dataset = load_ud_dataset(ud_path)

    if splits is None:
        splits = list(dataset.keys())

    kwargs = {}
    if analyzer == 'trmor':
        kwargs = {'morph_path': morph_path, 'fst_infl2_path': fst_infl2_path}
    elif analyzer == 'trmorph':
        kwargs = {
            'trmorph_fst_path': trmorph_fst_path,
            'flookup_path': flookup_path,
            'trmorph_mode': trmorph_mode,
        }
    elif analyzer == 'zemberek':
        kwargs = {}
    elif analyzer == 'zeyrek':
        kwargs = {}
    elif analyzer == 'vnlp':
        kwargs = {
            'vnlp_batch_size': vnlp_batch_size, 
            'vnlp_require_gpu': vnlp_require_gpu,
            'vnlp_memory_limit': vnlp_memory_limit
        }
        # Trigger VNLP/TensorFlow init and report GPU status
        from morphological_analyzer import vnlp as vnlp_mod
        _ = vnlp_mod.get_analyzer(batch_size=vnlp_batch_size, require_gpu=vnlp_require_gpu, memory_limit=vnlp_memory_limit)
        if _ is not None:
            print(vnlp_mod.get_vnlp_gpu_info(memory_limit=vnlp_memory_limit)[1])
    elif analyzer == 'smor':
        kwargs = {'smor_infl_path': smor_infl_path}
    elif analyzer == 'spacy_de':
        kwargs = {'model_name': spacy_model, 'spacy_require_gpu': spacy_require_gpu}

    analyzer_output_dir = os.path.join(output_dir, analyzer) if save_output else output_dir

    all_stats = {}

    for split in splits:
        if split not in dataset:
            print(f"⚠️  Split '{split}' not found, skipping...")
            continue

        sentences = dataset[split]
        if offset:
            sentences = sentences[offset:]
        if max_sentences:
            sentences = sentences[:max_sentences]

        print(f"\n🔬 Validating on {split} split ({len(sentences)} sentences) with {analyzer_label}...")

        results, stats = validate_on_dataset(
            sentences, analyzer, verbose=verbose, **kwargs
        )

        print_statistics(stats, split, analyzer_label=analyzer_label)
        all_stats[split] = stats

        if save_output:
            save_results(results, stats, analyzer_output_dir, split)

    if len(all_stats) > 1:
        print_combined_statistics(all_stats, analyzer_output_dir if save_output else None)

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a morphological analyser (TRMOR, TRmorph, ...) on UD Turkish-BOUN dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--ud-path",
        type=str,
        default=None,
        help="Path to UD dataset (default: data/UD_Turkish-BOUN, or data/UD_German-GSD when --analyzer smor or spacy_de)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=None,
        help="Splits to validate (train, dev, test). Default: all"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum sentences per split (for testing)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N sentences (for parallelizing datasets)"
    )
    parser.add_argument(
        "--analyzer",
        type=str,
        default="trmor",
        choices=ANALYZER_CHOICES,
        help="Morphological analyser to use"
    )
    parser.add_argument(
        "--morph-path",
        type=str,
        default="morphological_analyzer/TRMOR/morph.ca",
        help="Path to TRMOR morph.ca file (used when --analyzer trmor)"
    )
    parser.add_argument(
        "--fst-infl2-path",
        type=str,
        default="morphological_analyzer/SMOR/bin/fst-infl2",
        help="Path to fst-infl2 binary (used when --analyzer trmor)"
    )
    parser.add_argument(
        "--trmorph-fst",
        type=str,
        default="morphological_analyzer/TRmorph/trmorph.fst",
        help="Path to TRmorph FST file (used when --analyzer trmorph)"
    )
    parser.add_argument(
        "--flookup-path",
        type=str,
        default="flookup",
        help="Path to flookup binary (used when --analyzer trmorph)"
    )
    parser.add_argument(
        "--trmorph-mode",
        type=str,
        choices=("first", "majority"),
        default="first",
        help="TRmorph case prediction: 'first' = use only top analysis, 'majority' = majority vote over all analyses (used when --analyzer trmorph)"
    )
    parser.add_argument(
        "--smor-infl-path",
        type=str,
        default="morphological_analyzer/SMOR/smor-infl",
        help="Path to smor-infl script (used when --analyzer smor)"
    )
    parser.add_argument(
        "--vnlp-batch-size",
        type=int,
        default=128,
        help="VNLP predict batch_size (used when --analyzer vnlp). Increase for GPU speed; decrease if OOM."
    )
    parser.add_argument(
        "--vnlp-gpu",
        action="store_true",
        help="Require GPU for VNLP (TensorFlow). Fail if CUDA/GPU not visible. Use with sbatch --gres gpu:1."
    )
    parser.add_argument(
        "--vnlp-memory-limit",
        type=int,
        default=None,
        help="Limit GPU memory usage for VNLP (MB). Allows running many tasks in parallel on the same GPU."
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="de_dep_news_trf",
        help="spaCy German model to use when --analyzer spacy_de (e.g. de_dep_news_trf, de_core_news_lg)"
    )
    parser.add_argument(
        "--spacy-gpu",
        action="store_true",
        help="Require GPU for spaCy transformer model (fail if PyTorch has no CUDA). Use with sbatch when --gres gpu:1."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ud_validation_results",
        help="Directory to save validation results"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    start_time = time.time()

    main(
        ud_path=args.ud_path or ("data/UD_German-GSD" if args.analyzer in ("smor", "spacy_de") else "data/UD_Turkish-BOUN"),
        splits=args.splits,
        max_sentences=args.max_sentences,
        offset=args.offset,
        analyzer=args.analyzer,
        morph_path=args.morph_path,
        fst_infl2_path=args.fst_infl2_path,
        trmorph_fst_path=args.trmorph_fst,
        flookup_path=args.flookup_path,
        trmorph_mode=args.trmorph_mode,
        smor_infl_path=args.smor_infl_path,
        vnlp_batch_size=args.vnlp_batch_size,
        vnlp_require_gpu=args.vnlp_gpu,
        vnlp_memory_limit=args.vnlp_memory_limit,
        spacy_model=args.spacy_model,
        spacy_require_gpu=args.spacy_gpu,
        output_dir=args.output_dir,
        save_output=not args.no_save,
        verbose=not args.quiet
    )

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
