from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os
import re
from pathlib import Path


# Repo root: .../Morphological (this file lives in src/pretraining_dataset/)
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "data"


def _analysis_plot_path(filename: str) -> Path:
    """Return an absolute plot path under repo data/, creating the dir if needed."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR / filename


# Helper functions
def word_count(text):
    return len(text.split())

def char_count(text):
    return len(text)

def tokenize_text(text, lowercase=True):
    """
    Tokenize text by removing punctuation and optionally lowercasing.
    Filters out pure digit tokens.
    """
    # Remove punctuation (keep only word characters and whitespace)
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    words = cleaned.split()
    tokens = []
    for word in words:
        word = word.strip()
        if word and not word.isdigit():
            if lowercase:
                tokens.append(word.lower())
            else:
                tokens.append(word)
    return tokens


def analyze_dataset(dataset, dataset_name):
    """Analyze parallel corpus dataset and save visualizations."""
    print("Extracting data:")
    # Extract data
    en_texts = dataset['en']
    tr_texts = dataset['tr']
    
    print("Compute statistics:")
    # Compute statistics
    en_word_counts = [word_count(t) for t in en_texts]
    tr_word_counts = [word_count(t) for t in tr_texts]
    en_char_counts = [char_count(t) for t in en_texts]
    tr_char_counts = [char_count(t) for t in tr_texts]

    total_en_words = sum(en_word_counts)
    total_tr_words = sum(tr_word_counts)
    total_en_chars = sum(en_char_counts)
    total_tr_chars = sum(tr_char_counts)

    print("Vocabulary analysis:")
    # Vocabulary analysis (with proper tokenization - removes punctuation, filters digits)
    en_all_words = []
    for text in en_texts:
        en_all_words.extend(tokenize_text(text))
    tr_all_words = []
    for text in tr_texts:
        tr_all_words.extend(tokenize_text(text))
    en_vocab = set(en_all_words)
    tr_vocab = set(tr_all_words)
    en_ttr = len(en_vocab) / len(en_all_words) if en_all_words else 0
    tr_ttr = len(tr_vocab) / len(tr_all_words) if tr_all_words else 0

    print("Length ratios and correlation:")
    # Length ratios and correlation
    length_ratios = [en / tr if tr > 0 else 0 for en, tr in zip(en_char_counts, tr_char_counts)]
    length_ratios_valid = [r for r in length_ratios if r > 0]
    correlation = np.corrcoef(en_char_counts, tr_char_counts)[0, 1]

    print("Quality indicators:")
    # Quality indicators
    empty_en = sum(1 for t in en_texts if len(t.strip()) == 0)
    empty_tr = sum(1 for t in tr_texts if len(t.strip()) == 0)
    very_short_en = sum(1 for c in en_char_counts if c < 3)
    very_short_tr = sum(1 for c in tr_char_counts if c < 3)
    very_long_en = sum(1 for c in en_char_counts if c > 500)
    very_long_tr = sum(1 for c in tr_char_counts if c > 500)
    extreme_ratios = sum(1 for r in length_ratios_valid if r < 0.3 or r > 3.0)

    # ============================================================
    # VISUALIZATIONS (save only, no display)
    # ============================================================
    print("VISUALIZATIONS")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Character length distribution - English
    axes[0, 0].hist(en_char_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(en_char_counts), color='red', linestyle='--', label=f'Mean: {np.mean(en_char_counts):.1f}')
    axes[0, 0].axvline(np.median(en_char_counts), color='green', linestyle='--', label=f'Median: {np.median(en_char_counts):.1f}')
    axes[0, 0].set_xlabel('Character Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('English - Character Length Distribution')
    axes[0, 0].legend()

    # 2. Character length distribution - Turkish
    axes[0, 1].hist(tr_char_counts, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(np.mean(tr_char_counts), color='red', linestyle='--', label=f'Mean: {np.mean(tr_char_counts):.1f}')
    axes[0, 1].axvline(np.median(tr_char_counts), color='green', linestyle='--', label=f'Median: {np.median(tr_char_counts):.1f}')
    axes[0, 1].set_xlabel('Character Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Turkish - Character Length Distribution')
    axes[0, 1].legend()

    # 3. Word count distribution comparison
    axes[0, 2].hist(en_word_counts, bins=30, alpha=0.5, color='blue', label='English', edgecolor='black')
    axes[0, 2].hist(tr_word_counts, bins=30, alpha=0.5, color='orange', label='Turkish', edgecolor='black')
    axes[0, 2].set_xlabel('Words per Sentence')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Word Count Distribution (EN vs TR)')
    axes[0, 2].legend()

    # 4. Scatter plot: EN vs TR character lengths
    axes[1, 0].scatter(en_char_counts, tr_char_counts, alpha=0.1, s=1)
    axes[1, 0].plot([0, max(en_char_counts)], [0, max(en_char_counts)], 'r--', label='y=x')
    axes[1, 0].set_xlabel('English Character Length')
    axes[1, 0].set_ylabel('Turkish Character Length')
    axes[1, 0].set_title(f'EN vs TR Length (corr={correlation:.3f})')
    axes[1, 0].legend()

    # 5. Length ratio distribution
    axes[1, 1].hist(length_ratios_valid, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(1.0, color='red', linestyle='--', label='Ratio = 1.0')
    axes[1, 1].axvline(np.mean(length_ratios_valid), color='green', linestyle='--', label=f'Mean: {np.mean(length_ratios_valid):.2f}')
    axes[1, 1].set_xlabel('Length Ratio (EN/TR)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Source/Target Length Ratio Distribution')
    axes[1, 1].set_xlim(0, 5)
    axes[1, 1].legend()

    # 6. Box plots comparison
    box_data = [en_char_counts, tr_char_counts]
    bp = axes[1, 2].boxplot(box_data, labels=['English', 'Turkish'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('wheat')
    axes[1, 2].set_ylabel('Character Length')
    axes[1, 2].set_title('Character Length Box Plot')

    plt.tight_layout()
    plot_path = _analysis_plot_path(f"dataset_analysis_{dataset_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Close without displaying


    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("SUMMARY TABLE")
    summary_data = {
        'Metric': [
            'Total Samples',
            'Total Words',
            'Total Characters',
            'Vocabulary Size',
            'Type-Token Ratio',
            'Avg Words/Sentence',
            'Avg Chars/Sentence',
            'Median Chars/Sentence',
            'Std Dev (chars)',
            'Min Chars',
            'Max Chars',
            'Empty Sentences',
            'Very Short (<3 chars)',
            'Very Long (>500 chars)',
            'Potential Misalignments'
        ],
        'English': [
            f"{len(dataset):,}",
            f"{total_en_words:,}",
            f"{total_en_chars:,}",
            f"{len(en_vocab):,}",
            f"{en_ttr:.4f}",
            f"{np.mean(en_word_counts):.2f}",
            f"{np.mean(en_char_counts):.2f}",
            f"{np.median(en_char_counts):.2f}",
            f"{np.std(en_char_counts):.2f}",
            f"{np.min(en_char_counts)}",
            f"{np.max(en_char_counts)}",
            f"{empty_en:,}",
            f"{very_short_en:,}",
            f"{very_long_en:,}",
            f"{extreme_ratios:,} ({100*extreme_ratios/len(length_ratios_valid):.2f}%)"
        ],
        'Turkish': [
            f"{len(dataset):,}",
            f"{total_tr_words:,}",
            f"{total_tr_chars:,}",
            f"{len(tr_vocab):,}",
            f"{tr_ttr:.4f}",
            f"{np.mean(tr_word_counts):.2f}",
            f"{np.mean(tr_char_counts):.2f}",
            f"{np.median(tr_char_counts):.2f}",
            f"{np.std(tr_char_counts):.2f}",
            f"{np.min(tr_char_counts)}",
            f"{np.max(tr_char_counts)}",
            f"{empty_tr:,}",
            f"{very_short_tr:,}",
            f"{very_long_tr:,}",
            "-"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY: {dataset_name}")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))
    print(f"\nCorrelation (EN vs TR char length): {correlation:.4f}")
    print(f"Visualization saved to: {plot_path}")

    # ============================================================
    # SAMPLE DATA INSPECTION
    # ============================================================
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCE PAIRS")
    print("=" * 60)
    # Random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), 5, replace=False)

    for i, idx in enumerate(sample_indices):
        print(f"\n--- Sample {i+1} (idx={idx}) ---")
        print(f"EN: {dataset[int(idx)]['en']}")
        print(f"TR: {dataset[int(idx)]['tr']}")

    # Show some edge cases
    print("\n" + "=" * 60)
    print("EDGE CASES")
    print("=" * 60)

    # Shortest sentences
    shortest_en_idx = np.argmin(en_char_counts)
    shortest_tr_idx = np.argmin(tr_char_counts)
    longest_en_idx = np.argmax(en_char_counts)
    longest_tr_idx = np.argmax(tr_char_counts)

    print(f"\nShortest English sentence (idx={shortest_en_idx}):")
    print(f"  EN: '{dataset[int(shortest_en_idx)]['en']}'")
    print(f"  TR: '{dataset[int(shortest_en_idx)]['tr']}'")

    print(f"\nShortest Turkish sentence (idx={shortest_tr_idx}):")
    print(f"  EN: '{dataset[int(shortest_tr_idx)]['en']}'")
    print(f"  TR: '{dataset[int(shortest_tr_idx)]['tr']}'")

    print(f"\nLongest English sentence (idx={longest_en_idx}):")
    print(f"  EN: '{dataset[int(longest_en_idx)]['en'][:1000]}...'")
    print(f"  TR: '{dataset[int(longest_en_idx)]['tr'][:1000]}...'")

    print(f"\nLongest Turkish sentence (idx={longest_tr_idx}):")
    print(f"  EN: '{dataset[int(longest_tr_idx)]['en'][:1000]}...'")
    print(f"  TR: '{dataset[int(longest_tr_idx)]['tr'][:1000]}...'")


def analyze_single_language_dataset(dataset, dataset_name, language='en'):
    """
    Analyze a single-language dataset and save visualizations.
    
    Args:
        dataset: HuggingFace Dataset with 'text' column (or 'en'/'tr' column)
        dataset_name: Name for output files
        language: Language code ('en' for English, 'tr' for Turkish)
    """
    # Determine the text column name
    if 'text' in dataset.column_names:
        text_column = 'text'
    elif language in dataset.column_names:
        text_column = language
    else:
        raise ValueError(f"Dataset must have 'text' or '{language}' column. Found: {dataset.column_names}")
    
    # Language display names and colors
    lang_info = {
        'en': {'name': 'English', 'color': 'blue', 'light_color': 'lightblue'},
        'tr': {'name': 'Turkish', 'color': 'orange', 'light_color': 'wheat'},
        'de': {'name': 'German', 'color': 'green', 'light_color': 'lightgreen'},
        'deu': {'name': 'German', 'color': 'green', 'light_color': 'lightgreen'},
        'eng': {'name': 'English', 'color': 'blue', 'light_color': 'lightblue'},
        'tur': {'name': 'Turkish', 'color': 'orange', 'light_color': 'wheat'},
    }
    
    lang_name = lang_info.get(language, {'name': language.upper(), 'color': 'gray', 'light_color': 'lightgray'})['name']
    lang_color = lang_info.get(language, {'name': language.upper(), 'color': 'gray', 'light_color': 'lightgray'})['color']
    lang_light_color = lang_info.get(language, {'name': language.upper(), 'color': 'gray', 'light_color': 'lightgray'})['light_color']
    
    print(f"Analyzing {lang_name} dataset: {dataset_name}")
    # Extract data
    print("Extracting data...")
    texts = dataset[text_column]
    
    # Compute statistics
    word_counts = [word_count(t) for t in texts]
    char_counts = [char_count(t) for t in texts]
    
    total_words = sum(word_counts)
    total_chars = sum(char_counts)
    
    # Vocabulary analysis (with proper tokenization - removes punctuation, filters digits)
    print("Vocabulary analysis...")
    all_words = []
    for text in texts:
        all_words.extend(tokenize_text(text))
    vocab = set(all_words)
    ttr = len(vocab) / len(all_words) if len(all_words) > 0 else 0
    
    # Quality indicators
    empty_count = sum(1 for t in texts if len(t.strip()) == 0)
    very_short = sum(1 for c in char_counts if c < 3)
    very_long = sum(1 for c in char_counts if c > 500)
    
    # Word frequency analysis
    print("Word frequency analysis...")
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(20)
    
    # ============================================================
    # VISUALIZATIONS (save only, no display)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Character length distribution
    axes[0, 0].hist(char_counts, bins=50, alpha=0.7, color=lang_color, edgecolor='black')
    axes[0, 0].axvline(np.mean(char_counts), color='red', linestyle='--', label=f'Mean: {np.mean(char_counts):.1f}')
    axes[0, 0].axvline(np.median(char_counts), color='green', linestyle='--', label=f'Median: {np.median(char_counts):.1f}')
    axes[0, 0].set_xlabel('Character Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{lang_name} - Character Length Distribution')
    axes[0, 0].legend()
    
    # 2. Word count distribution
    axes[0, 1].hist(word_counts, bins=50, alpha=0.7, color=lang_color, edgecolor='black')
    axes[0, 1].axvline(np.mean(word_counts), color='red', linestyle='--', label=f'Mean: {np.mean(word_counts):.1f}')
    axes[0, 1].axvline(np.median(word_counts), color='green', linestyle='--', label=f'Median: {np.median(word_counts):.1f}')
    axes[0, 1].set_xlabel('Words per Sentence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{lang_name} - Word Count Distribution')
    axes[0, 1].legend()
    
    # 3. Top 20 words bar chart
    if top_words:
        words, counts = zip(*top_words)
        axes[0, 2].barh(range(len(words)), counts, color=lang_color, edgecolor='black')
        axes[0, 2].set_yticks(range(len(words)))
        axes[0, 2].set_yticklabels(words)
        axes[0, 2].invert_yaxis()
        axes[0, 2].set_xlabel('Frequency')
        axes[0, 2].set_title(f'{lang_name} - Top 20 Words')
    
    # 4. Log-scale character length distribution
    axes[1, 0].hist(char_counts, bins=50, alpha=0.7, color=lang_color, edgecolor='black', log=True)
    axes[1, 0].set_xlabel('Character Length')
    axes[1, 0].set_ylabel('Frequency (log scale)')
    axes[1, 0].set_title(f'{lang_name} - Character Length (Log Scale)')
    
    # 5. Cumulative distribution
    sorted_chars = np.sort(char_counts)
    cumulative = np.arange(1, len(sorted_chars) + 1) / len(sorted_chars)
    axes[1, 1].plot(sorted_chars, cumulative, color=lang_color, linewidth=2)
    axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    axes[1, 1].axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    axes[1, 1].set_xlabel('Character Length')
    axes[1, 1].set_ylabel('Cumulative Proportion')
    axes[1, 1].set_title(f'{lang_name} - Cumulative Distribution')
    axes[1, 1].legend()
    
    # 6. Box plot
    bp = axes[1, 2].boxplot([char_counts], labels=[lang_name], patch_artist=True)
    bp['boxes'][0].set_facecolor(lang_light_color)
    axes[1, 2].set_ylabel('Character Length')
    axes[1, 2].set_title(f'{lang_name} - Character Length Box Plot')
    
    plt.tight_layout()
    output_path = _analysis_plot_path(f"dataset_analysis_{dataset_name}_{language}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("Generating summary...")
    summary_data = {
        'Metric': [
            'Total Samples',
            'Total Words',
            'Total Characters',
            'Vocabulary Size',
            'Type-Token Ratio',
            'Avg Words/Sentence',
            'Avg Chars/Sentence',
            'Median Chars/Sentence',
            'Std Dev (chars)',
            'Min Chars',
            'Max Chars',
            '25th Percentile (chars)',
            '75th Percentile (chars)',
            '90th Percentile (chars)',
            'Empty Sentences',
            'Very Short (<3 chars)',
            'Very Long (>500 chars)',
        ],
        lang_name: [
            f"{len(dataset):,}",
            f"{total_words:,}",
            f"{total_chars:,}",
            f"{len(vocab):,}",
            f"{ttr:.4f}",
            f"{np.mean(word_counts):.2f}",
            f"{np.mean(char_counts):.2f}",
            f"{np.median(char_counts):.2f}",
            f"{np.std(char_counts):.2f}",
            f"{np.min(char_counts)}",
            f"{np.max(char_counts)}",
            f"{np.percentile(char_counts, 25):.2f}",
            f"{np.percentile(char_counts, 75):.2f}",
            f"{np.percentile(char_counts, 90):.2f}",
            f"{empty_count:,}",
            f"{very_short:,}",
            f"{very_long:,}",
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY: {dataset_name} ({lang_name})")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))
    print(f"\nVisualization saved to: {output_path}")
    
    # ============================================================
    # TOP WORDS
    # ============================================================
    print("\n" + "=" * 60)
    print(f"TOP 20 WORDS ({lang_name})")
    print("=" * 60)
    for i, (word, count) in enumerate(top_words, 1):
        print(f"  {i:2d}. {word}: {count:,}")
    
    # ============================================================
    # SAMPLE DATA INSPECTION
    # ============================================================
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES")
    print("=" * 60)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n--- Sample {i+1} (idx={idx}) ---")
        text = dataset[int(idx)][text_column]
        print(f"  {text[:500]}{'...' if len(text) > 500 else ''}")
    
    # Edge cases
    print("\n" + "=" * 60)
    print("EDGE CASES")
    print("=" * 60)
    
    shortest_idx = np.argmin(char_counts)
    longest_idx = np.argmax(char_counts)
    
    print(f"\nShortest sentence (idx={shortest_idx}, {char_counts[shortest_idx]} chars):")
    print(f"  '{dataset[int(shortest_idx)][text_column]}'")
    
    print(f"\nLongest sentence (idx={longest_idx}, {char_counts[longest_idx]} chars):")
    print(f"  '{dataset[int(longest_idx)][text_column][:1000]}...'")
    
    return {
        'total_samples': len(dataset),
        'total_words': total_words,
        'total_chars': total_chars,
        'vocab_size': len(vocab),
        'ttr': ttr,
        'avg_words': np.mean(word_counts),
        'avg_chars': np.mean(char_counts),
        'median_chars': np.median(char_counts),
    }