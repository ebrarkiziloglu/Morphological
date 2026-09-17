import argparse
import os
import subprocess
import json
import math
import sys
import time

# Import helper functions from morphological_analysis_helpers
try:
    from morphological_analysis_helpers import print_statistics, combine_statistics, parse_conllu_file
except ImportError:
    print("Error: morphological_analysis_helpers.py not found in the current directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run analyze_UD.py in parallel chunks and merge results for VNLP.")
    parser.add_argument("--ud-path", type=str, default="data/UD_Turkish-BOUN", help="Path to UD dataset directory.")
    parser.add_argument("--num-chunks", type=int, default=32, help="Number of chunks to split each dataset split into.")
    parser.add_argument("--vnlp-memory-limit", type=int, default=512, help="Memory limit per VNLP instance in MB.")
    parser.add_argument("--vnlp-batch-size", type=int, default=128, help="Batch size for VNLP predictions.")
    parser.add_argument("--skip-run", action="store_true", help="Skip running analyze_UD.py and only perform merging (if results already exist).")
    args = parser.parse_args()

    start_time = time.time()

    splits_to_process = [
        ("train", "tr_boun-ud-train.conllu"),
        ("dev", "tr_boun-ud-dev.conllu"),
        ("test", "tr_boun-ud-test.conllu")
    ]
    
    all_splits_stats = {}

    for split_name, split_filename in splits_to_process:
        print(f"\n" + "="*80)
        print(f"PROCESSING SPLIT: {split_name.upper()}")
        print("="*80)
        
        split_file = os.path.join(args.ud_path, split_filename)
        if not os.path.exists(split_file):
            print(f"Error: dataset file not found at {split_file}")
            continue

        print(f"📁 Reading UD dataset from {split_file}...")
        sentences = parse_conllu_file(split_file)
        total_sentences = len(sentences)
        print(f"Loaded {split_name} set with {total_sentences} sentences.")

        num_chunks = args.num_chunks
        chunk_size = math.ceil(total_sentences / num_chunks)
        print(f"Splitting {total_sentences} sentences into {num_chunks} tasks (chunk size: {chunk_size})")

        processes = []

        if not args.skip_run:
            print(f"\n🚀 Starting parallel execution of analyze_UD.py for {split_name}...")
            for i in range(num_chunks):
                offset = i * chunk_size
                if offset >= total_sentences:
                    break
                    
                output_dir = f"ud_validation_results/vnlp_part_{split_name}_{i}"
                
                # Command to run for each chunk. Use sys.executable to ensure we use the current python environment.
                cmd = [
                    sys.executable, "analyze_UD.py",
                    "--analyzer", "vnlp",
                    "--vnlp-batch-size", str(args.vnlp_batch_size),
                    "--vnlp-memory-limit", str(args.vnlp_memory_limit),
                    "--vnlp-gpu",
                    "--splits", split_name,
                    "--offset", str(offset),
                    "--max-sentences", str(chunk_size),
                    "--output-dir", output_dir,
                    "--quiet"
                ]
                
                print(f"   Spawning {split_name} chunk {i} (offset {offset}, max sentences {chunk_size}) -> {output_dir}")
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processes.append((i, p))
                
            print(f"\n⏳ Waiting for all {split_name} chunks to finish processing...")
            for i, p in processes:
                p.wait()
                
            print(f"✅ All parallel tasks for {split_name} completed.\n")
        else:
            print(f"\n⏩ Skipping execution for {split_name}, proceeding directly to merging...")

        print(f"📈 Merging and printing {split_name} chunk statistics...")
        all_stats = {}
        
        for i in range(num_chunks):
            stats_file = f"ud_validation_results/vnlp_part_{split_name}_{i}/vnlp/{split_name}_stats.json"
            
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    chunk_name = f"chunk_{i}"
                    all_stats[chunk_name] = stats
            else:
                pass

        if all_stats:
            combined_stats = combine_statistics(all_stats)
            
            print("\n\n" + "-" * 80)
            print(f"          FINAL MERGED 📊 PER-CASE PERFORMANCE RESULTS FOR {split_name.upper()}")
            print("-" * 80)
            print_statistics(combined_stats, split_name=f"MERGED_{split_name.upper()}", analyzer_label=f"VNLP ({split_name})")
            
            merged_dir = "ud_validation_results/vnlp_merged"
            os.makedirs(merged_dir, exist_ok=True)
            merged_file = os.path.join(merged_dir, f"{split_name}_stats.json")
            with open(merged_file, 'w', encoding='utf-8') as f:
                json.dump(combined_stats, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Saved overall merged {split_name} statistics to {merged_file}")
            
            all_splits_stats[split_name] = combined_stats
        else:
            print(f"❌ No statistics files were found to merge for {split_name}. Check if the analyze_UD.py runs succeeded.")

    if all_splits_stats:
        from morphological_analysis_helpers import print_combined_statistics
        print("\n\n" + "="*80)
        print("FINAL MERGED RESULTS ACROSS ALL SPLITS")
        print("="*80)
        print_combined_statistics(all_splits_stats, output_dir="ud_validation_results/vnlp_merged")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n⏱️  Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
