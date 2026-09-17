import os
import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datasets import load_dataset, Dataset
from tqdm import tqdm

def get_log_normal_pdf(x, mu, sigma):
    if x <= 0:
        return 0.0
    return (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(-((math.log(x) - mu)**2) / (2 * sigma**2))

def extract_german_subset(output_path, original_bytes, original_mean, original_median, original_std, byte_premium, token=None):
    target_bytes = int(original_bytes * byte_premium)
    target_mean = original_mean * byte_premium
    target_median = original_median * byte_premium
    target_std = original_std * byte_premium

    print(f"Byte premium: {byte_premium}")
    print(f"{'Metric':<20}{'Original':>15}{'Target':>15}")
    print("-" * 50)
    print(f"{'Bytes':<20}{original_bytes:>15,}{target_bytes:>15,}")
    print(f"{'Mean':<20}{original_mean:>15,.2f}{target_mean:>15,.2f}")
    print(f"{'Median':<20}{original_median:>15,.2f}{target_median:>15,.2f}")
    print(f"{'Std':<20}{original_std:>15,.2f}{target_std:>15,.2f}")

    print(f"Loading German BabyLM dataset (checking local cache)...")
    
    local_dir = "data/babylm-deu"
    
    # Check if local dataset exists
    if not os.path.exists(local_dir):
        print(f"Downloading German BabyLM dataset to {local_dir}...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="BabyLM-community/babylm-deu", repo_type="dataset", local_dir=local_dir, token=token)
        
    print(f"Loading German BabyLM dataset from local dir ({local_dir})...")
    dataset = load_dataset(
        local_dir, 
        split="train"
    )
    
    # Target distribution parameters based on English pretraining set scaled by 1.0537
    mu = math.log(target_median)
    sigma = math.sqrt(2 * math.log(target_mean / target_median))
    
    # 1. Collect a large pool of valid sentences and group by length bin (10 char width)
    print("Collecting candidate sentences to match the target length distribution...")
    pool_bins = defaultdict(list)
    pool_total_bytes = 0
    
    import nltk
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    
    # We want to collect ~3x our target bytes to have a rich pool for distribution shaping
    for item in tqdm(dataset, desc="Buffering candidates"):
        if 'text' in item:
            text = item['text']
        elif 'sentence' in item:
            text = item['sentence']
        else:
            text = next((v for v in item.values() if isinstance(v, str)), "")
            
        if not isinstance(text, str):
            continue
            
        text = text.replace("\n", " ").strip()
        
        # Split paragraph into sentences
        sentences = nltk.sent_tokenize(text, language="german")
        
        for sent in sentences:
            char_len = len(sent)
            word_count = len(sent.split())
            
            # Apply min/max bounds so pool matches target (median ~203, mean ~230 chars)
            # Require at least 20 words and 100 chars to avoid short sentences pulling mean down
            if word_count < 20 or char_len < 100 or char_len > 1500:
                continue
                
            line = sent + "\n"
            line_bytes = len(line.encode("utf-8"))
            
            bin_idx = char_len // 10
            pool_bins[bin_idx].append((line, char_len, line_bytes))
            pool_total_bytes += line_bytes
            
        # Stop collecting when we have a pool 5x the target bytes (larger pool
        # keeps target-length bins from exhausting so achieved distribution matches better)
        if pool_total_bytes > target_bytes * 5:
            break
            
    print(f"\nCollected a pool of {pool_total_bytes / 1024 / 1024:.2f} MB. Now sampling...")
    
    # 2. Compute the ideal probability for each bin using the Log-Normal PDF
    bin_probs = {}
    for bin_idx in pool_bins.keys():
        bin_center = (bin_idx * 10) + 5
        bin_probs[bin_idx] = get_log_normal_pdf(bin_center, mu, sigma)
        
    # 3. Sample sentences bin by bin according to probabilities
    sampled_lines = []
    sampled_chars = []
    current_bytes = 0
    
    # Pre-shuffle the bins so we can pop randomly
    for bin_idx in pool_bins:
        random.shuffle(pool_bins[bin_idx])
        
    available_bins = list(pool_bins.keys())
    
    with tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Sampling matches") as pbar:
        while current_bytes < target_bytes and available_bins:
            # Re-normalize probabilities based on remaining available bins
            current_total_prob = sum(bin_probs[b] for b in available_bins)
            if current_total_prob == 0:
                break
                
            # Roulette wheel selection
            r = random.random() * current_total_prob
            cum_prob = 0
            selected_bin = available_bins[-1]
            
            for b in available_bins:
                cum_prob += bin_probs[b]
                if r <= cum_prob:
                    selected_bin = b
                    break
                    
            # Pop a sentence from the selected bin
            line, char_len, line_bytes = pool_bins[selected_bin].pop()
            sampled_lines.append(line)
            sampled_chars.append(char_len)
            
            current_bytes += line_bytes
            pbar.update(line_bytes)
            
            # Remove bin if it is empty
            if not pool_bins[selected_bin]:
                available_bins.remove(selected_bin)
                
    # 4. Save to file
    random.shuffle(sampled_lines)  # Shuffle so the data isn't artificially ordered
    
    print(f"\nSaving exactly {current_bytes:,} bytes to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)
        
    # 5. Print statistics to verify
    mean_chars = statistics.mean(sampled_chars) if sampled_chars else 0
    median_chars = statistics.median(sampled_chars) if sampled_chars else 0
    std_chars = statistics.stdev(sampled_chars) if len(sampled_chars) > 1 else 0
    
    print(f"\n=== SAMPLING COMPLETE ===")
    print(f"Total samples written: {len(sampled_lines):,}")
    print(f"Actual bytes written:  {current_bytes:,}")
    print(f"Target bytes were:     {target_bytes:,}")
    print(f"Difference:            {current_bytes - target_bytes:,} bytes")
    
    print("\n--- Achieved Length Distribution (Characters) ---")
    print(f"Mean:   {mean_chars:.2f} (Target: ~{target_mean:.2f})")
    print(f"Median: {median_chars:.2f} (Target: ~{target_median:.2f})")
    print(f"StdDev: {std_chars:.2f} (Target: ~{target_std:.2f})")
    print(f"Min:    {min(sampled_chars)}")
    print(f"Max:    {max(sampled_chars)}")
    print(f"\nSaved locally to: {output_path}")


def analyze_german_cases(dataset_path: str, spacy_model: str = "de_core_news_lg"):
    print(f"\n--- Starting morphological analysis using spaCy ({spacy_model}) ---")
    try:
        import spacy
    except ImportError:
        print("spaCy is not installed. Please install it using 'pip install spacy'.")
        return

    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        print(f"Model '{spacy_model}' not found. Downloading...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
        nlp = spacy.load(spacy_model)

    print(f"Reading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    case_counts = Counter()
    total_tokens = 0
    total_words_with_case = 0

    print(f"Analyzing {len(lines)} sentences...")
    
    # Disable components not needed for morphological analysis to speed up processing
    # 'tok2vec', 'tagger', 'morphologizer' are kept.
    disable_pipes = ["parser", "ner", "attribute_ruler", "lemmatizer"]
    
    if "trf" in spacy_model:
        spacy.prefer_gpu()

    # Process sentences in batches for speed
    for doc in tqdm(nlp.pipe(lines, batch_size=1000, disable=disable_pipes), total=len(lines), desc="Analyzing Cases"):
        for token in doc:
            total_tokens += 1
            cases = token.morph.get("Case")
            if cases:
                case = cases[0]  # Take the first case
                case_counts[case] += 1
                total_words_with_case += 1

    print("\n=== Case Occurrences ===")
    for case, count in case_counts.most_common():
        percentage = (count / total_words_with_case) * 100 if total_words_with_case > 0 else 0
        print(f"{case}: {count:,} ({percentage:.2f}% of words with case)")

    print(f"\nTotal tokens analyzed: {total_tokens:,}")
    print(f"Total tokens with case: {total_words_with_case:,}")

    # Save statistics to a JSON file
    results = {
        "total_tokens": total_tokens,
        "total_tokens_with_case": total_words_with_case,
        "case_counts": dict(case_counts),
        "case_percentages": {case: (count / total_words_with_case) * 100 for case, count in case_counts.items()}
    }

    stats_path = dataset_path + ".case_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nCase statistics saved to {stats_path}")


def analyze_german_distribution(dataset_path: str, dataset_name: str = "BabyLM_DE_subset_de"):
    """
    Analyze the German subset using the generic dataset analysis helper.
    Produces plots and a textual summary similar to the English/HPLT analysis.
    """
    from dataset_analysis_helper import analyze_single_language_dataset

    print(f"\n--- Analyzing distributional properties of {dataset_path} ---")
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    hf_dataset = Dataset.from_dict({"text": lines})
    analyze_single_language_dataset(hf_dataset, dataset_name=dataset_name, language="de")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and analyze a subset of the German BabyLM dataset")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token for gated dataset")
    parser.add_argument("--analyze-only", action="store_true", help="Skip extraction and only run spaCy analysis")
    parser.add_argument("--spacy-model", type=str, default="de_core_news_lg", help="spaCy German model to use")
    parser.add_argument("--original-bytes", type=int, required=True, help="Original bytes for the subset")
    parser.add_argument("--original-mean", type=float, required=True, help="Original mean for the subset")
    parser.add_argument("--original-median", type=float, required=True, help="Original median for the subset")
    parser.add_argument("--original-std", type=float, required=True, help="Original std for the subset")
    parser.add_argument("--byte-premium", type=float, required=True, help="Byte premium for the subset")
    
    args = parser.parse_args()
    
    output_dir = "data/BabyLM_DE_subset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "BabyLM_DE_subset.de")
    
    if not args.analyze_only:
        extract_german_subset(
            output_path, 
            args.original_bytes, 
            args.original_mean, 
            args.original_median, 
            args.original_std, 
            args.byte_premium, 
            token=args.token)

    analyze_german_cases(dataset_path=output_path, spacy_model=args.spacy_model)
    analyze_german_distribution(dataset_path=output_path, dataset_name="BabyLM_DE_subset_de")