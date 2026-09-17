import argparse
import os
import subprocess
import json
import math
import sys
import time
from collections import defaultdict
import tempfile
import re

try:
    from datasets import load_from_disk, Dataset
except ImportError:
    print("Warning: datasets library not found. --tr-path loading from arrow will fail if used.")

# ============================================================
# 1. Word Extraction & Case Assignment Helpers
# ============================================================

def extract_words_from_sentences(sentences, track_sentences=False, max_examples=3):
    unique_words = set()
    word_to_sentences = defaultdict(list) if track_sentences else None
    word_frequencies = defaultdict(int)
    
    for sentence in sentences:
        cleaned = re.sub(r'[^\w\s]', ' ', sentence)
        words = cleaned.split()
        for word in words:
            word = word.strip()
            if word and not word.isdigit():
                word_lower = word.lower()
                unique_words.add(word_lower)
                word_frequencies[word_lower] += 1
                
                if track_sentences and len(word_to_sentences[word_lower]) < max_examples:
                    word_to_sentences[word_lower].append(sentence)
    
    if track_sentences:
        return list(unique_words), dict(word_to_sentences), dict(word_frequencies)
    return list(unique_words), dict(word_frequencies)

CASE_SUFFIX_RULES = {
    'accusative': {'ı', 'i', 'u', 'ü'},          
    'dative': {'a', 'e'},                        
    'genitive': {'ın', 'in', 'un', 'ün'},        
    'locative': {'da', 'de', 'ta', 'te'},        
    'ablative': {'dan', 'den', 'tan', 'ten'},        
    'nominative': None,                          
}

def has_valid_case_suffix(word, case_name):
    expected_suffixes = CASE_SUFFIX_RULES.get(case_name)
    if expected_suffixes is None:
        return True
    if not word:
        return False
    word_lower = word.lower()
    for suffix in expected_suffixes:
        if len(word_lower) >= len(suffix) and word_lower.endswith(suffix):
            return True
    return False

def parse_vnlp_analysis(analysis):
    from morphological_analyzer.vnlp import extract_case_from_vnlp
    case = extract_case_from_vnlp(analysis)
    return "<" + case + ">" if case else None

def parse_zemberek_analysis(analysis):
    from morphological_analyzer.zemberek import extract_case_from_zemberek
    case = extract_case_from_zemberek(analysis)
    return "<" + case + ">" if case else None

def analyze_cases(analyses, analyzer_name, word_frequencies):
    # Analyze the parsed dictionaries for case markers
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
        case_counts = defaultdict(int)

        for analysis in word_analyses:
            if analyzer_name == 'vnlp':
                c = parse_vnlp_analysis(analysis)
            elif analyzer_name == 'zemberek':
                c = parse_zemberek_analysis(analysis)
            else:
                c = None

            if c:
                c_name = None
                for name, tag in case_markers:
                    if tag == c:
                        c_name = name
                        break
                if c_name:
                    case_counts[c_name] += 1
        
        if not case_counts:
            results['no_case'].add(word)
            continue
            
        max_count = max(case_counts.values())
        winning_case = None
        for case_name, _ in case_markers:
            if case_counts.get(case_name, 0) == max_count:
                winning_case = case_name
                break

        is_regular = has_valid_case_suffix(word, winning_case)

        results[winning_case][word] = {
            'count': max_count,
            'total': total_analyses,
            'ratio': max_count / total_analyses,
            'is_regular': is_regular,
            'frequency': word_frequencies.get(word, 1)
        }

    return results

# ============================================================
# 2. Worker Parallel Tools
# ============================================================

def worker_main(args):
    with open(args.worker_input, 'r', encoding='utf-8') as f:
        words = json.load(f)
        
    print(f"[Worker] Starting {args.analyzer} on chunk of {len(words)} words...")
    
    if args.analyzer == 'zemberek':
        from morphological_analyzer.zemberek import run_zemberek_analyser
        # For a single word-list chunk, run zemberek directly. 
        # (It will load Java within this worker process, do the job, and save it)
        analyses_dict, no_res = run_zemberek_analyser(words, use_disambiguation=False, max_workers=1)
    elif args.analyzer == 'vnlp':
        from morphological_analyzer.vnlp import run_vnlp_analyser
        # For VNLP we also load TF into this worker process and execute
        analyses_dict, no_res = run_vnlp_analyser(words, batch_size=args.vnlp_batch_size, require_gpu=args.vnlp_gpu, memory_limit=args.vnlp_memory_limit)
    else:
        raise ValueError(f"Unsupported analyzer {args.analyzer}")
        
    output = {
        'analyses': analyses_dict,
        'no_result': list(no_res)
    }
    with open(args.worker_output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
    
    print(f"[Worker] Finished chunk and saved to {args.worker_output}")

# ============================================================
# 3. Print and Save Formatting
# ============================================================

def print_statistics(unique_words, analyses, case_results, word_frequencies, no_analysis_words=None):
    print("=" * 60)
    print("MORPHOLOGICAL ANALYSIS STATISTICS")
    print("=" * 60)

    total_unique = len(unique_words)
    total_occurrences = sum(word_frequencies.values())
    
    analyzed_unique = len(analyses)
    analyzed_occ = sum(word_frequencies.get(w, 1) for w in analyses)
    
    if no_analysis_words is not None:
        no_analysis_unique = len(no_analysis_words)
        no_analysis_occ = sum(word_frequencies.get(w, 1) for w in no_analysis_words)
    else:
        no_analysis_unique = len(case_results.get('no_analysis', set()))
        no_analysis_occ = sum(word_frequencies.get(w, 1) for w in case_results.get('no_analysis', set()))
        
    no_case_set = case_results.get('no_case', set())
    no_case_unique = len(no_case_set)
    no_case_occ = sum(word_frequencies.get(w, 1) for w in no_case_set)

    case_names = ['accusative', 'dative', 'genitive', 'locative', 'ablative', 'nominative']
    case_tags = {'accusative': '<Acc>', 'dative': '<Dat>', 'genitive': '<Gen>',
                 'locative': '<Loc>', 'ablative': '<Abl>', 'nominative': '<Nom>'}
    
    print(f"\n📊 Basic Statistics:")
    print(f"   Total occurrences:               {total_occurrences:,}")
    print(f"   Total unique words:              {total_unique:,}")
    print(f"   Words with analysis:             {analyzed_occ:,} occurrences ({analyzed_unique:,} unique)")
    print(f"   Words without analysis:          {no_analysis_occ:,} occurrences ({no_analysis_unique:,} unique)")
    print(f"   Words without case markers:      {no_case_occ:,} occurrences ({no_case_unique:,} unique)")

    total_with_case = sum(sum(s.get('frequency', 1) for s in case_results[c].values()) for c in case_names)

    counts, percentages = [], []
    regular_counts, irregular_counts = [], []
    regular_ratios, avg_ratios, max_ratios, min_ratios = [], [], [], []

    for case_name in case_names:
        case_data = case_results[case_name]
        count = sum(s.get('frequency', 1) for s in case_data.values())
        counts.append(f"{count:,}")

        if total_with_case > 0:
            pct = count / total_with_case * 100
            percentages.append(f"{pct:.2f}%")
        else:
            percentages.append("N/A")

        if case_data:
            regular = sum(s.get('frequency', 1) for s in case_data.values() if s.get('is_regular', True))
            irregular = count - regular
            regular_counts.append(f"{regular:,}")
            irregular_counts.append(f"{irregular:,}")
            regular_ratios.append(f"{regular/count:.4f}" if count > 0 else "N/A")

            weighted_ratio_sum = sum(s['ratio'] * s.get('frequency', 1) for s in case_data.values())
            avg_ratio = weighted_ratio_sum / count if count > 0 else 0
            
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

    print(f"\n📊 Case Analysis Summary Table (Based on Word Occurrences):\n")
    print(f"| Metric                  | Acc       | Dat       | Gen       | Loc       | Abl       | Nom       |")
    print(f"|-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|")
    print(f"| Occurrences Dist        | {counts[0]:>9} | {counts[1]:>9} | {counts[2]:>9} | {counts[3]:>9} | {counts[4]:>9} | {counts[5]:>9} |")
    print(f"| Percentage              | {percentages[0]:>9} | {percentages[1]:>9} | {percentages[2]:>9} | {percentages[3]:>9} | {percentages[4]:>9} | {percentages[5]:>9} |")
    print(f"| Regular Occurrences     | {regular_counts[0]:>9} | {regular_counts[1]:>9} | {regular_counts[2]:>9} | {regular_counts[3]:>9} | {regular_counts[4]:>9} | {regular_counts[5]:>9} |")
    print(f"| Irregular Occurrences   | {irregular_counts[0]:>9} | {irregular_counts[1]:>9} | {irregular_counts[2]:>9} | {irregular_counts[3]:>9} | {irregular_counts[4]:>9} | {irregular_counts[5]:>9} |")
    print(f"| Regular Ratio           | {regular_ratios[0]:>9} | {regular_ratios[1]:>9} | {regular_ratios[2]:>9} | {regular_ratios[3]:>9} | {regular_ratios[4]:>9} | {regular_ratios[5]:>9} |")
    print(f"| Avg Case Ratio          | {avg_ratios[0]:>9} | {avg_ratios[1]:>9} | {avg_ratios[2]:>9} | {avg_ratios[3]:>9} | {avg_ratios[4]:>9} | {avg_ratios[5]:>9} |")
    print(f"| Max Case Ratio          | {max_ratios[0]:>9} | {max_ratios[1]:>9} | {max_ratios[2]:>9} | {max_ratios[3]:>9} | {max_ratios[4]:>9} | {max_ratios[5]:>9} |")
    print(f"| Min Case Ratio          | {min_ratios[0]:>9} | {min_ratios[1]:>9} | {min_ratios[2]:>9} | {min_ratios[3]:>9} | {min_ratios[4]:>9} | {min_ratios[5]:>9} |")
    print("\n" + "=" * 60)

def save_results(case_results, word_frequencies, no_analysis_words=None, output_dir="morphological_analysis", word_to_sentences=None):
    case_suffix_display = {
        'accusative': '-ı, -i, -u, -ü',
        'dative': '-a, -e',
        'genitive': '-n',
        'locative': '-a, -e',
        'ablative': '-n',
        'nominative': '(no specific suffix)',
    }

    def sort_by_ratio_desc(items):
        return sorted(items, key=lambda x: (-x[1]['ratio'], -x[1].get('frequency', 1), x[0]))

    def write_markdown_table(f, title, case_data, include_header=True, include_context=False):
        if include_header:
            f.write(f"## {title}\n\n")
            
        total_unique = len(case_data)
        total_occ = sum(s.get('frequency', 1) for s in case_data.values())
        f.write(f"**Total occurrences:** {total_occ:,}\n")
        f.write(f"**Total unique words:** {total_unique:,}\n\n")
        
        if not case_data:
            f.write("*No words in this category*\n\n")
            return
            
        f.write("*Sorted by ratio then frequency (descending)*\n\n")
        f.write("| Word | Occurrences | Analyses Count | Total Analyses | Ratio |\n")
        f.write("|------|-------------|----------------|----------------|-------|\n")
        for word, stats in sort_by_ratio_desc(case_data.items()):
            f.write(f"| {word} | {stats.get('frequency', 1):,} | {stats['count']} | {stats['total']} | {stats['ratio']:.4f} |\n")
        f.write("\n")
        if include_context and word_to_sentences:
            f.write("### Context Sentences for Irregular Words\n\n")
            for word, stats in sort_by_ratio_desc(case_data.items()):
                if word in word_to_sentences and word_to_sentences[word]:
                    f.write(f"**{word}** (ratio: {stats['ratio']:.4f} | occurrences: {stats.get('frequency', 1):,})\n\n")
                    for i, sentence in enumerate(word_to_sentences[word], 1):
                        escaped_sentence = sentence.replace('|', '\\|')
                        f.write(f"{i}. {escaped_sentence}\n")
                    f.write("\n")
            f.write("\n")

    cases = [
        ('accusative', '<Acc>'),
        ('dative', '<Dat>'),
        ('genitive', '<Gen>'),
        ('locative', '<Loc>'),
        ('ablative', '<Abl>'),
        ('nominative', '<Nom>'),
    ]

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    for case_name, case_tag in cases:
        filename = f"{case_name}_words.md"
        filepath = os.path.join(output_dir, filename)
        case_data = case_results[case_name]
        expected_suffix = case_suffix_display[case_name]
        
        regular_words = {w: s for w, s in case_data.items() if s.get('is_regular', True)}
        irregular_words = {w: s for w, s in case_data.items() if not s.get('is_regular', True)}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Words with {case_name.capitalize()} case (`{case_tag}`)\n\n")
            
            f.write(f"**Total occurrences:** {sum(s.get('frequency',1) for s in case_data.values()):,}\n")
            f.write(f"**Total unique words:** {len(case_data):,}\n\n")
            f.write(f"**Expected suffix:** {expected_suffix}\n\n")
            f.write("*Each word is assigned to exactly one case (the one with the highest count)*\n\n")
            f.write("---\n\n")
            write_markdown_table(f, f"Regular Words (ending with {expected_suffix})", regular_words)
            f.write("---\n\n")
            write_markdown_table(f, f"Irregular Words (NOT ending with {expected_suffix})", irregular_words, include_context=True)
        saved_files.append(f"   - {output_dir}/{filename}")

    if no_analysis_words:
        no_analysis_filepath = os.path.join(output_dir, "no_analysis_words.md")
        sorted_words = sorted(no_analysis_words, key=lambda w: (-word_frequencies.get(w, 1), w))
        with open(no_analysis_filepath, 'w', encoding='utf-8') as f:
            f.write("# Words with No Morphological Analysis\n\n")
            f.write(f"**Total occurrences:** {sum(word_frequencies.get(w, 1) for w in no_analysis_words):,}\n")
            f.write(f"**Total unique words:** {len(no_analysis_words):,}\n\n")
            f.write("---\n\n")
            f.write("## Word List\n\n")
            f.write("*Sorted by occurrence (descending)*\n\n")
            f.write("| # | Word | Occurrences |\n")
            f.write("|---|------|-------------|\n")
            for i, word in enumerate(sorted_words, 1):
                f.write(f"| {i} | {word} | {word_frequencies.get(word, 1):,} |\n")
        saved_files.append(f"   - {output_dir}/no_analysis_words.md")

    print(f"\n✅ Results saved to:")
    for file_path in saved_files:
        print(file_path)

# ============================================================
# 4. Main Parallel Orchestration
# ============================================================

def main_orchestrator(args):
    print(f"🚀 Starting parallel morphological analysis ({args.analyzer}) pipeline on {args.dataset_name}...")
    start_time = time.time()

    print("\n📁 Step 1: Loading dataset...")
    dataset = load_from_disk(args.tr_path)
    print(" ** 1 ** Dataset has splits: {dataset}")

    if 'tr' in dataset.column_names:
        dataset = dataset['tr']
    
    # Check if nested (DatasetDict), unnest to base Dataset
    if hasattr(dataset, "keys") and callable(getattr(dataset, "keys", None)):
        if 'tr' in dataset: dataset = dataset['tr']
        elif 'train' in dataset: dataset = dataset['train']
        elif 'text' in dataset: dataset = dataset['text']
        else: dataset = dataset[list(dataset.keys())[0]]

    total_sents = len(dataset)
    num_to_process = min(args.max_samples, total_sents) if args.max_samples else total_sents
    
    # 🚨 SLICE FIRST! 🚨 to avoid evaluating all 38 Million sentences from PyArrow!
    data_dict = dataset[:num_to_process]
    
    if 'tr' in data_dict:
        dataset_slice = data_dict['tr']
    elif 'text' in data_dict: 
        dataset_slice = data_dict['text']
    else: 
        try:
            dataset_slice = data_dict[list(data_dict.keys())[0]]
        except:
            dataset_slice = data_dict

    print(f"   Loaded {num_to_process} sentences for processing (out of {total_sents} total)")

    print("\n🔤 Step 2: Extracting unique Turkish words...")
    unique_words, word_to_sentences, word_frequencies = extract_words_from_sentences(dataset_slice, track_sentences=True, max_examples=3)
    num_unique = len(unique_words)
    print(f"   Found {num_unique:,} unique words.")

    print(f"\n🔬 Step 3: Running morphological analyser {args.analyzer} in {args.num_chunks} parallel chunks...")
    
    chunk_size = math.ceil(num_unique / args.num_chunks)
    temp_dir = tempfile.mkdtemp()
    print(f"   Created temporary workspace: {temp_dir}")
    print(f"   Chunk size: {chunk_size} words per chunk")
    
    processes = []
    
    for i in range(args.num_chunks):
        offset = i * chunk_size
        if offset >= num_unique:
            break
            
        chunk_words = unique_words[offset:offset+chunk_size]
        in_file = os.path.join(temp_dir, f"chunk_in_{i}.json")
        out_file = os.path.join(temp_dir, f"chunk_out_{i}.json")
        
        with open(in_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_words, f, ensure_ascii=False)
            
        cmd = [
            sys.executable, __file__,
            "--worker-mode",
            "--worker-input", in_file,
            "--worker-output", out_file,
            "--analyzer", args.analyzer,
            "--vnlp-batch-size", str(args.vnlp_batch_size),
        ]
        
        if args.vnlp_gpu:
            cmd.append("--vnlp-gpu")
        if args.vnlp_memory_limit:
            cmd.extend(["--vnlp-memory-limit", str(args.vnlp_memory_limit)])
            
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
        processes.append((i, p, out_file))
        
    print(f"   Spawned {len(processes)} tasks. Waiting for completion...")
    
    for i, p, out_file in processes:
        p.wait()
        
    print(f"✅ All parallel chunks processed.")
    
    print("\n📈 Step 4: Merging results...")
    all_analyses = {}
    all_no_result = set()
    
    for i, p, out_file in processes:
        if os.path.exists(out_file):
            with open(out_file, 'r', encoding='utf-8') as f:
                res = json.load(f)
                all_analyses.update(res['analyses'])
                all_no_result.update(res['no_result'])
        else:
            print(f"   ⚠️ Warning: Chunk {i} output file not found! Worker may have crashed.")

    print(f"   Merged. Successfully analyzed {len(all_analyses):,} words.")
    print(f"   Words with no analysis: {len(all_no_result):,}")

    print("\n📑 Step 5: Analyzing grammatical cases...")
    case_results = analyze_cases(all_analyses, args.analyzer, word_frequencies)
    
    print_statistics(unique_words, all_analyses, case_results, word_frequencies, all_no_result)

    if not args.no_save:
        out_dir = os.path.join(args.output_dir, args.dataset_name + f"_{args.analyzer}")
        save_results(case_results, word_frequencies, all_no_result, out_dir, word_to_sentences)

    end_time = time.time()
    samples_info = f"{num_to_process} sentences"
    print(f"\n⏱️  Total time taken: {end_time - start_time:.2f} seconds for {samples_info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Dataset Analyser for Zemberek & VNLP")
    
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-input", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=str, help=argparse.SUPPRESS)
    
    parser.add_argument("--tr-path", type=str, default="data/arrow/HPLT_en_tr", help="Path to Arrow dataset")
    parser.add_argument("--dataset_name", type=str, default="HPLT_en_tr", help="Name of the dataset")
    parser.add_argument("--max-samples", type=int, default=100000, help="Max sentences to read")
    parser.add_argument("--num-chunks", type=int, default=32, help="Number of chunks to parallelize into")
    
    parser.add_argument("--analyzer", type=str, required=False, default="zemberek", choices=["zemberek", "vnlp"], help="Which morphological analyzer to use")
    
    parser.add_argument("--vnlp-batch-size", type=int, default=128)
    parser.add_argument("--vnlp-gpu", action="store_true")
    parser.add_argument("--vnlp-memory-limit", type=int, default=512)
    
    parser.add_argument("--output-dir", type=str, default="morphological_analysis_hplt")
    parser.add_argument("--no-save", action="store_true")
    
    args = parser.parse_args()
    
    if args.worker_mode:
        worker_main(args)
    else:
        main_orchestrator(args)
