"""
Turkish Morphological Analysis Script

This script:
1. Loads a parallel EN-TR dataset
2. Extracts unique Turkish words
3. Analyzes each word with the Turkish morphological analyser (fst-infl2)
4. Identifies grammatical cases: Accusative (<Acc>), Dative (<Dat>), Genitive (<Gen>),
   Locative (<Loc>), Ablative (<Abl>), and Nominative (<Nom>)
5. Assigns each word to exactly one case group (the one with the highest analysis count)
6. Reports statistics
"""

import subprocess
import tempfile
import os
import re
import sys
import argparse
from collections import defaultdict
import time
from datasets import Dataset, load_from_disk

# ============================================================
# 2. Morphological Analyser Interface
# ============================================================

def extract_words_from_sentences(sentences, track_sentences=False, max_examples=3):
    """Extract unique words from a list of sentences.
    
    Args:
        sentences: List of sentence strings
        track_sentences: If True, also returns a mapping of word -> example sentences
        max_examples: Maximum number of example sentences to store per word (default: 3)
    
    Returns:
        If track_sentences=False: set of unique words
        If track_sentences=True: tuple (set of unique words, dict {word: [sentences]})
    """
    unique_words = set()
    word_to_sentences = defaultdict(list) if track_sentences else None
    
    # Simple tokenization: split by whitespace and remove punctuation
    for sentence in sentences:
        # Remove common punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', sentence)
        words = cleaned.split()
        for word in words:
            word = word.strip()
            if word and not word.isdigit():
                word_lower = word.lower()
                unique_words.add(word_lower)
                
                # Track example sentences for this word
                if track_sentences and len(word_to_sentences[word_lower]) < max_examples:
                    word_to_sentences[word_lower].append(sentence)
    
    if track_sentences:
        return unique_words, dict(word_to_sentences)
    return unique_words


def run_morphological_analyser(words, morph_path="morphological_analyzer/TRMOR/morph.ca", fst_infl2_path="morphological_analyzer/SMOR/bin/fst-infl2"):
    """
    Run the Turkish morphological analyser on a list of words.
    
    Args:
        words: Iterable of words to analyze
        morph_path: Path to the TRMOR/morph.ca file
        fst_infl2_path: Path to fst-infl2 binary
    
    Returns:
        tuple: (dict {word: list of analyses}, set of words with no analysis)
    """
    # Create a temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')
        input_file = f.name

    # Create a temporary output file
    output_file = input_file.replace('.txt', '_output.txt')

    try:
        # Run the morphological analyser
        # Command: fst-infl2 TRMOR/morph.ca input.txt > output.txt
        cmd = f"{fst_infl2_path} {morph_path} {input_file}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # Check if the command failed
        if result.returncode != 0:
            print(f"Error: morphological analyser command failed with exit code {result.returncode}")
            print(f"Command: {cmd}")
            print(f"Stderr: {result.stderr}")
            return {}, set()
        
        # Check if we got any output
        if not result.stdout.strip():
            print(f"Warning: morphological analyser produced no output")
            print(f"Command: {cmd}")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            return {}, set()

        output = result.stdout

        # Parse the output
        analyses, no_result_words = parse_analyser_output(output)
        return analyses, no_result_words
        
    except Exception as e:
        print(f"Error running morphological analyser: {e}")
        if 'result' in locals() and result.stderr:
            print(f"Error output: {result.stderr}")
        return {}, set()

    finally:
        # Clean up temporary files
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)


def parse_analyser_output(output):
    """
    Parse the output of the morphological analyser.
    
    Output format:
    > word
    analysis1
    analysis2
    ...
    > nextword
    no result for nextword
    ...
    
    Returns:
        tuple: (dict {word: list of analyses}, set of words with no analysis)
    """
    analyses = defaultdict(list)
    no_result_words = set()
    current_word = None

    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('> '):
            # New word
            current_word = line[2:].strip()
        elif line.startswith('no result for '):
            # No analysis available - extract and track the word
            word = line[len('no result for '):].strip()
            no_result_words.add(word)
        elif current_word is not None:
            # This is an analysis for the current word
            analyses[current_word].append(line)

    return analyses, no_result_words


def tokenize_sentence(sentence, lowercase=True):
    """
    Tokenize a sentence into words.
    
    Args:
        sentence: A sentence string
        lowercase: Whether to lowercase the words (default: True)
    
    Returns:
        list: List of tokens (words)
    """
    # Remove common punctuation and split
    cleaned = re.sub(r'[^\w\s]', ' ', sentence)
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


def analyze_sentence(sentence, morph_path="morphological_analyzer/TRMOR/morph.ca", fst_infl2_path="morphological_analyzer/SMOR/bin/fst-infl2", 
                     lowercase=True, preserve_order=False):
    """
    Run morphological analysis on a single sentence.
    
    Args:
        sentence: A sentence string to analyze
        morph_path: Path to the TRMOR/morph.ca file
        fst_infl2_path: Path to fst-infl2 binary
        lowercase: Whether to lowercase words before analysis (default: True)
        preserve_order: If True, returns a list of (word, analyses) tuples in sentence order;
                        If False, returns a dict {word: analyses} (default: False)
    
    Returns:
        If preserve_order=True: list of (word, list of analyses) tuples in sentence order
        If preserve_order=False: dict {word: list of analyses}
    
    Example:
        >>> analyses = analyze_sentence("Kitabı okudum.")
        >>> print(analyses)
        {'kitabı': ['kitap<N><sg><Acc>', ...], 'okudum': ['oku<V><past><1sg>', ...]}
        
        >>> analyses = analyze_sentence("Kitabı okudum.", preserve_order=True)
        >>> print(analyses)
        [('kitabı', ['kitap<N><sg><Acc>', ...]), ('okudum', ['oku<V><past><1sg>', ...])]
    """
    # Tokenize the sentence
    tokens = tokenize_sentence(sentence, lowercase=lowercase)
    
    if not tokens:
        return [] if preserve_order else {}
    
    # Run the morphological analyser on all tokens
    analyses, _ = run_morphological_analyser(tokens, morph_path=morph_path, fst_infl2_path=fst_infl2_path)
    
    if preserve_order:
        # Return list of tuples in sentence order (may have duplicates if word appears multiple times)
        return [(token, analyses.get(token, [])) for token in tokens]
    else:
        return dict(analyses)


def analyze_sentences(sentences, morph_path="morphological_analyzer/TRMOR/morph.ca", fst_infl2_path="morphological_analyzer/SMOR/bin/fst-infl2",
                      lowercase=True, unique_only=True):
    """
    Run morphological analysis on multiple sentences.
    
    This is more efficient than calling analyze_sentence() multiple times because
    it batches all unique words and runs the analyser only once.
    
    Args:
        sentences: List of sentence strings to analyze
        morph_path: Path to the TRMOR/morph.ca file
        fst_infl2_path: Path to fst-infl2 binary
        lowercase: Whether to lowercase words before analysis (default: True)
        unique_only: If True, returns analyses for unique words only (dict);
                     If False, returns per-sentence analyses (list of dicts)
    
    Returns:
        If unique_only=True: dict {word: list of analyses} for all unique words
        If unique_only=False: list of dicts, one per sentence
    
    Example:
        >>> sentences = ["Kitabı okudum.", "Eve gittim."]
        >>> analyses = analyze_sentences(sentences)
        >>> print(analyses.keys())
        dict_keys(['kitabı', 'okudum', 'eve', 'gittim'])
        
        >>> analyses = analyze_sentences(sentences, unique_only=False)
        >>> print(len(analyses))  # 2 (one per sentence)
    """
    # Collect all tokens from all sentences
    all_tokens = []
    sentence_tokens = []  # Keep track of tokens per sentence for unique_only=False
    
    for sentence in sentences:
        tokens = tokenize_sentence(sentence, lowercase=lowercase)
        all_tokens.extend(tokens)
        sentence_tokens.append(tokens)
    
    if not all_tokens:
        return {} if unique_only else [{} for _ in sentences]
    
    # Get unique words for efficient analysis
    unique_words = set(all_tokens)
    
    # Run the morphological analyser once on all unique words
    analyses, _ = run_morphological_analyser(unique_words, morph_path=morph_path, fst_infl2_path=fst_infl2_path)
    
    if unique_only:
        return dict(analyses)
    else:
        # Return per-sentence analyses
        result = []
        for tokens in sentence_tokens:
            sentence_analyses = {token: analyses.get(token, []) for token in set(tokens)}
            result.append(sentence_analyses)
        return result


# ============================================================
# 3. Case Analysis Functions
# ============================================================

# Expected suffixes for each case (Turkish vowel harmony)
CASE_SUFFIX_RULES = {
    'accusative': {'ı', 'i', 'u', 'ü'},          # -ı, -i, -u, -ü
    'dative': {'a', 'e'},                        # -a, -e
    'genitive': {'ın', 'in', 'un', 'ün'},        # -n (actually -ın, -in, -un, -ün but ends with n)
    'locative': {'da', 'de', 'ta', 'te'},        # -da/-de/-ta/-te (ends with a or e)
    'ablative': {'dan', 'den', 'tan', 'ten'},        # -dan/-den/-tan/-ten (ends with n)
    'nominative': None,                          # No specific suffix (bare stem)
}


def has_valid_case_suffix(word, case_name):
    """
    Check if a word has a valid suffix for the given case.
    
    Args:
        word: The word to check
        case_name: The case name (accusative, dative, etc.)
    
    Returns:
        bool: True if the word has a valid suffix for the case, False otherwise.
              Returns True for cases with no suffix requirement (nominative).
    """
    expected_suffixes = CASE_SUFFIX_RULES.get(case_name)
    
    if expected_suffixes is None:
        # No suffix requirement (e.g., nominative)
        return True

    if not word:
        return False

    # Check for each possible suffix (could be multiple letters)
    word_lower = word.lower()
    for suffix in expected_suffixes:
        if len(word_lower) >= len(suffix) and word_lower.endswith(suffix):
            return True
    return False


def analyze_cases(analyses):
    """
    Analyze the morphological analyses for case markers.
    
    Each noun is assigned to exactly ONE case group based on which case marker
    has the highest count among its analyses. In case of a tie, priority order is:
    accusative > dative > genitive > locative > ablative > nominative
    
    Additionally, each word is checked for valid case suffixes:
    - Accusative: must end with -ı, -i, -u, -ü
    - Dative/Locative: must end with -a or -e
    - Genitive/Ablative: must end with -n
    - Nominative: no specific suffix requirement
    
    Args:
        analyses: dict {word: list of analyses}
    
    Returns:
        dict with case information:
        {
            'accusative': {word: {'count': N, 'total': T, 'ratio': N/T, 'is_regular': bool}, ...},
            'dative': {word: {'count': N, 'total': T, 'ratio': N/T, 'is_regular': bool}, ...},
            'genitive': {word: {'count': N, 'total': T, 'ratio': N/T, 'is_regular': bool}, ...},
            'locative': {word: {'count': N, 'total': T, 'ratio': N/T, 'is_regular': bool}, ...},
            'ablative': {word: {'count': N, 'total': T, 'ratio': N/T, 'is_regular': bool}, ...},
            'nominative': {word: {'count': N, 'total': T, 'ratio': N/T, 'is_regular': bool}, ...},
            'no_analysis': set of words with no analysis,
            'no_case': set of words with analysis but no case markers
        }
    """
    # Case markers and their corresponding result keys
    # Order determines priority in case of tie (first = highest priority)
    case_markers = [
        ('accusative', '<Acc>'),
        ('dative', '<Dat>'),
        ('genitive', '<Gen>'),
        ('locative', '<Loc>'),
        ('ablative', '<Abl>'),
        ('nominative', '<Nom>'),
    ]
    
    results = {case_name: {} for case_name, _ in case_markers}
    results['no_analysis'] = set()
    results['no_case'] = set()

    for word, word_analyses in analyses.items():
        if not word_analyses:
            results['no_analysis'].add(word)
            continue

        total_analyses = len(word_analyses)

        # Count analyses containing each case marker
        case_counts = {}
        for case_name, marker in case_markers:
            count = sum(1 for a in word_analyses if marker in a)
            if count > 0:
                case_counts[case_name] = count

        if not case_counts:
            # Word has analyses but none with case markers
            results['no_case'].add(word)
            continue

        # Find the case with the highest count
        # In case of tie, the order in case_markers determines priority
        max_count = max(case_counts.values())
        
        # Get the winning case (first one with max_count, respecting priority order)
        winning_case = None
        for case_name, _ in case_markers:
            if case_counts.get(case_name, 0) == max_count:
                winning_case = case_name
                break

        # Check if word has valid suffix for the assigned case
        is_regular = has_valid_case_suffix(word, winning_case)

        # Assign word to the winning case group
        results[winning_case][word] = {
            'count': max_count,
            'total': total_analyses,
            'ratio': max_count / total_analyses,
            'is_regular': is_regular
        }

    return results


# ============================================================
# 4. Statistics and Reporting
# ============================================================

def print_statistics(unique_words, analyses, case_results, no_analysis_words=None):
    """Print statistics about the morphological analysis."""

    print("=" * 60)
    print("MORPHOLOGICAL ANALYSIS STATISTICS")
    print("=" * 60)

    total_unique = len(unique_words)
    analyzed = len(analyses)
    # Use no_analysis_words if provided, otherwise fall back to case_results
    if no_analysis_words is not None:
        no_analysis = len(no_analysis_words)
    else:
        no_analysis = len(case_results.get('no_analysis', set()))
    no_case = len(case_results.get('no_case', set()))

    # All case names in display order
    case_names = ['accusative', 'dative', 'genitive', 'locative', 'ablative', 'nominative']
    case_tags = {'accusative': '<Acc>', 'dative': '<Dat>', 'genitive': '<Gen>',
                 'locative': '<Loc>', 'ablative': '<Abl>', 'nominative': '<Nom>'}
    case_suffixes = {'accusative': '-ı/-i/-u/-ü', 'dative': '-a/-e', 'genitive': '-n',
                     'locative': '-a/-e', 'ablative': '-n', 'nominative': '(any)'}

    print(f"\n📊 Basic Statistics:")
    print(f"   Total unique words:              {total_unique:,}")
    print(f"   Words with analysis:             {analyzed:,}")
    print(f"   Words without analysis:          {no_analysis:,}")
    print(f"   Words without case markers:      {no_case:,}")

    # Calculate total words assigned to cases
    total_with_case = sum(len(case_results[c]) for c in case_names)

    # Prepare data for table
    counts = []
    percentages = []
    regular_counts = []
    irregular_counts = []
    regular_ratios = []
    avg_ratios = []
    max_ratios = []
    min_ratios = []

    for case_name in case_names:
        case_data = case_results[case_name]
        count = len(case_data)
        counts.append(f"{count:,}")

        # Percentage
        if total_with_case > 0:
            pct = count / total_with_case * 100
            percentages.append(f"{pct:.2f}%")
        else:
            percentages.append("N/A")

        # Regular/Irregular counts and ratios
        if case_data:
            regular = sum(1 for s in case_data.values() if s.get('is_regular', True))
            irregular = count - regular
            regular_counts.append(f"{regular:,}")
            irregular_counts.append(f"{irregular:,}")
            regular_ratios.append(f"{regular/count:.4f}" if count > 0 else "N/A")

            avg_ratio = sum(s['ratio'] for s in case_data.values()) / len(case_data)
            min_ratio = min(s['ratio'] for s in case_data.values())
            max_ratio = max(s['ratio'] for s in case_data.values())
            avg_ratios.append(f"{avg_ratio:.4f}")
            min_ratios.append(f"{min_ratio:.4f}")
            max_ratios.append(f"{max_ratio:.4f}")
        else:
            regular_counts.append("N/A")
            irregular_counts.append("N/A")
            regular_ratios.append("N/A")
            avg_ratios.append("N/A")
            min_ratios.append("N/A")
            max_ratios.append("N/A")

    # Print Markdown table
    print(f"\n📊 Case Analysis Summary Table:\n")
    print(f"| Metric                  | Acc       | Dat       | Gen       | Loc       | Abl       | Nom       |")
    print(f"|-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|")
    print(f"| Case Distribution       | {counts[0]:>9} | {counts[1]:>9} | {counts[2]:>9} | {counts[3]:>9} | {counts[4]:>9} | {counts[5]:>9} |")
    print(f"| Percentage              | {percentages[0]:>9} | {percentages[1]:>9} | {percentages[2]:>9} | {percentages[3]:>9} | {percentages[4]:>9} | {percentages[5]:>9} |")
    print(f"| Regular Count           | {regular_counts[0]:>9} | {regular_counts[1]:>9} | {regular_counts[2]:>9} | {regular_counts[3]:>9} | {regular_counts[4]:>9} | {regular_counts[5]:>9} |")
    print(f"| Irregular Count         | {irregular_counts[0]:>9} | {irregular_counts[1]:>9} | {irregular_counts[2]:>9} | {irregular_counts[3]:>9} | {irregular_counts[4]:>9} | {irregular_counts[5]:>9} |")
    print(f"| Regular Ratio           | {regular_ratios[0]:>9} | {regular_ratios[1]:>9} | {regular_ratios[2]:>9} | {regular_ratios[3]:>9} | {regular_ratios[4]:>9} | {regular_ratios[5]:>9} |")
    print(f"| Avg Case Ratio          | {avg_ratios[0]:>9} | {avg_ratios[1]:>9} | {avg_ratios[2]:>9} | {avg_ratios[3]:>9} | {avg_ratios[4]:>9} | {avg_ratios[5]:>9} |")
    print(f"| Max Case Ratio          | {max_ratios[0]:>9} | {max_ratios[1]:>9} | {max_ratios[2]:>9} | {max_ratios[3]:>9} | {max_ratios[4]:>9} | {max_ratios[5]:>9} |")
    print(f"| Min Case Ratio          | {min_ratios[0]:>9} | {min_ratios[1]:>9} | {min_ratios[2]:>9} | {min_ratios[3]:>9} | {min_ratios[4]:>9} | {min_ratios[5]:>9} |")

    print("\n" + "=" * 60)


def save_results(case_results, no_analysis_words=None, output_dir="morphological_analysis", word_to_sentences=None):
    """Save the case analysis results to markdown files, sorted by ratio in descending order.
    
    Each case file contains two tables:
    1. Regular words (with expected suffix)
    2. Irregular words (without expected suffix) - with context sentences if available
    
    Args:
        case_results: Dict with case analysis results
        no_analysis_words: Set of words with no morphological analysis
        output_dir: Directory to save results
        word_to_sentences: Optional dict {word: [example sentences]} for context
    """

    # Expected suffixes for display
    case_suffix_display = {
        'accusative': '-ı, -i, -u, -ü',
        'dative': '-a, -e',
        'genitive': '-n',
        'locative': '-a, -e',
        'ablative': '-n',
        'nominative': '(no specific suffix)',
    }

    def sort_by_ratio_desc(items):
        """Sort items by ratio (descending), then alphabetically for ties."""
        return sorted(items, key=lambda x: (-x[1]['ratio'], x[0]))

    def write_markdown_table(f, title, case_data, include_header=True, include_context=False):
        """Write a markdown table for case results.
        
        Args:
            f: File handle
            title: Section title
            case_data: Dict {word: stats}
            include_header: Whether to write the section header
            include_context: Whether to include context sentences (for irregular words)
        """
        if include_header:
            f.write(f"## {title}\n\n")
        
        f.write(f"**Total words:** {len(case_data)}\n\n")
        
        if not case_data:
            f.write("*No words in this category*\n\n")
            return

        f.write("*Sorted by ratio (descending)*\n\n")

        # Table header
        f.write("| Word | Count | Total Analyses | Ratio |\n")
        f.write("|------|-------|----------------|-------|\n")

        # Table rows
        for word, stats in sort_by_ratio_desc(case_data.items()):
            f.write(f"| {word} | {stats['count']} | {stats['total']} | {stats['ratio']:.4f} |\n")
        
        f.write("\n")
        
        # Add context sentences for irregular words if available
        if include_context and word_to_sentences:
            f.write("### Context Sentences for Irregular Words\n\n")
            for word, stats in sort_by_ratio_desc(case_data.items()):
                if word in word_to_sentences and word_to_sentences[word]:
                    f.write(f"**{word}** (ratio: {stats['ratio']:.4f})\n\n")
                    for i, sentence in enumerate(word_to_sentences[word], 1):
                        # Escape any pipe characters in the sentence
                        escaped_sentence = sentence.replace('|', '\\|')
                        f.write(f"{i}. {escaped_sentence}\n")
                    f.write("\n")
            f.write("\n")

    # All cases to save
    cases = [
        ('accusative', '<Acc>'),
        ('dative', '<Dat>'),
        ('genitive', '<Gen>'),
        ('locative', '<Loc>'),
        ('ablative', '<Abl>'),
        ('nominative', '<Nom>'),
    ]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    for case_name, case_tag in cases:
        filename = f"{case_name}_words.md"
        filepath = os.path.join(output_dir, filename)
        
        case_data = case_results[case_name]
        expected_suffix = case_suffix_display[case_name]
        
        # Separate regular and irregular words
        regular_words = {w: s for w, s in case_data.items() if s.get('is_regular', True)}
        irregular_words = {w: s for w, s in case_data.items() if not s.get('is_regular', True)}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Words with {case_name.capitalize()} case (`{case_tag}`)\n\n")
            f.write(f"**Total words:** {len(case_data)}\n\n")
            f.write(f"**Expected suffix:** {expected_suffix}\n\n")
            f.write("*Each word is assigned to exactly one case (the one with the highest count)*\n\n")
            f.write("---\n\n")
            
            # Regular words table
            write_markdown_table(f, f"Regular Words (ending with {expected_suffix})", regular_words)
            
            f.write("---\n\n")
            
            # Irregular words table (with context sentences if available)
            write_markdown_table(f, f"Irregular Words (NOT ending with {expected_suffix})", irregular_words, include_context=True)
        
        saved_files.append(f"   - {output_dir}/{filename}")

    # Save words with no analysis
    if no_analysis_words:
        no_analysis_filepath = os.path.join(output_dir, "no_analysis_words.md")
        sorted_words = sorted(no_analysis_words)
        
        with open(no_analysis_filepath, 'w', encoding='utf-8') as f:
            f.write("# Words with No Morphological Analysis\n\n")
            f.write(f"**Total words:** {len(no_analysis_words):,}\n\n")
            f.write("These words were not recognized by the Turkish morphological analyser (TRMOR).\n\n")
            f.write("Possible reasons:\n")
            f.write("- Foreign words (English, Arabic, etc.)\n")
            f.write("- Proper nouns not in the lexicon\n")
            f.write("- Misspellings or typos\n")
            f.write("- Neologisms or slang\n")
            f.write("- Non-standard spellings\n\n")
            f.write("---\n\n")
            f.write("## Word List\n\n")
            f.write("*Sorted alphabetically*\n\n")
            f.write("| # | Word |\n")
            f.write("|---|------|\n")
            
            for i, word in enumerate(sorted_words, 1):
                f.write(f"| {i} | {word} |\n")
        
        saved_files.append(f"   - {output_dir}/no_analysis_words.md")

    print(f"\n✅ Results saved to:")
    for file_path in saved_files:
        print(file_path)


# ============================================================
# 5. Main Pipeline
# ============================================================

def main(
    tr_path,
    load_from_arrow=False,
    max_samples=None,
    morph_path="morphological_analyzer/TRMOR/morph.ca",
    fst_infl2_path="morphological_analyzer/SMOR/bin/fst-infl2",
    output_dir="morphological_analysis",
    save_output=True
):
    """
    Main pipeline for morphological analysis.
    
    Args:
        tr_path: Path to Turkish sentences file
        max_samples: Maximum number of sentences to process (None for all)
        morph_path: Path to the TRMOR/morph.ca file
        fst_infl2_path: Path to fst-infl2 binary
        output_dir: Directory to save results
        save_output: Whether to save results to files
    """

    print("🚀 Starting morphological analysis pipeline...")

    # Step 1: Load dataset
    print("\n📁 Step 1: Loading dataset...")
    if load_from_arrow:
        dataset = load_from_disk(tr_path)
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    else:
        raise ValueError(f"Do not support loading dataset from {tr_path}")

    if isinstance(dataset, dict) and 'tr' in dataset:
        print(f"   Dataset is a dictionary and 'tr' is in dataset")
        dataset = dataset['tr']
    elif isinstance(dataset, dict) and 'text' in dataset:
        print(f"   Dataset is a dictionary and 'text' is in dataset")
        dataset = dataset['text']
    elif isinstance(dataset, Dataset):
        print(f"   Dataset is a Dataset")
        dataset = dataset.to_dict()
        if 'text' in dataset:
            dataset = dataset['text']
        else:
            print(f"   ### Dataset keys are: {dataset.keys()}")
    else:
        raise ValueError("Invalid dataset type")
    # print(f"Dataset: {dataset}")
    # print(f"Dataset keys: {dataset.keys()}")
    # print(f"Dataset type: {type(dataset)}")
    # print(f"Dataset length: {len(dataset)}")
    
    num_samples = len(dataset)
    print(f"   Loaded {num_samples} sentence pairs")

    # Step 2: Extract unique Turkish words (with sentence tracking for context)
    print("\n🔤 Step 2: Extracting unique Turkish words...")
    unique_words, word_to_sentences = extract_words_from_sentences(dataset, track_sentences=True, max_examples=3)
    print(f"   Found {len(unique_words):,} unique words")
    print(f"   Tracking example sentences for context")

    # Step 3: Run morphological analyser
    print("\n🔬 Step 3: Running morphological analyser...")
    print(f"   This may take a while for large datasets...")
    analyses, no_analysis_words = run_morphological_analyser(
        unique_words,
        morph_path=morph_path,
        fst_infl2_path=fst_infl2_path
    )
    print(f"   Analyzed {len(analyses):,} words")
    print(f"   Words with no analysis: {len(no_analysis_words):,}")

    # Step 4: Analyze cases
    print("\n📑 Step 4: Analyzing grammatical cases...")
    case_results = analyze_cases(analyses)
    print(f"Case keys: {case_results.keys()}")
    # Step 5: Print statistics
    print_statistics(unique_words, analyses, case_results, no_analysis_words)

    # Step 6: Save results (optional)
    if save_output:
        save_results(case_results, no_analysis_words, output_dir, word_to_sentences)

    return {
        'unique_words': unique_words,
        'analyses': analyses,
        'no_analysis_words': no_analysis_words,
        'case_results': case_results,
        'word_to_sentences': word_to_sentences
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turkish Morphological Analysis Script - Analyzes Turkish words for grammatical cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    

    parser.add_argument(
        "--tr-path",
        type=str,
        default="data/arrow/HPLT_en_tr/HPLT.en-tr.tr",
        help="Path to Turkish sentences file"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="OpenSubtitles_2024_en_tr",
        help="Name of the dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of sentences to process (None for all)"
    )
    parser.add_argument(
        "--arrow",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--morph-path",
        type=str,
        default="morphological_analyzer/TRMOR/morph.ca",
        help="Path to the TRMOR/morph.ca file"
    )
    parser.add_argument(
        "--fst-infl2-path",
        type=str,
        default="morphological_analyzer/SMOR/bin/fst-infl2",
        help="Path to fst-infl2 binary"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="morphological_analysis",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to files"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    results = main(
        tr_path=args.tr_path,
        load_from_arrow=args.arrow,
        max_samples=args.max_samples,
        morph_path=args.morph_path,
        fst_infl2_path=args.fst_infl2_path,
        output_dir=args.output_dir+"/"+args.dataset_name,
        save_output=not args.no_save
    )
    
    end_time = time.time()
    samples_info = f"{args.max_samples} samples" if args.max_samples else "all samples"
    print(f"Time taken: {end_time - start_time:.4f} seconds for {samples_info}")
