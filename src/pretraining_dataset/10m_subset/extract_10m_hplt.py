from datasets import load_from_disk, Dataset
from analyze_pretraining_dataset import load_moses_chunked
from dataset_analysis_helper import analyze_dataset, word_count
import os

def extract_10m_subset():
    # 1. Load HPLT dataset
    print("Loading HPLT dataset...")
    
    # Check if we already have the cached arrow format, so we don't need to load from raw text
    arrow_path = "data/arrow/HPLT_en_tr"
    if os.path.exists(arrow_path):
        print(f"Loading directly from arrow format: {arrow_path}")
        hplt_dataset = load_from_disk(arrow_path)
    else:
        print("Arrow format not found, loading from moses chunks...")
        hplt_dataset = load_moses_chunked(
            "data/HPLT_en_tr/HPLT.en-tr.en",
            "data/HPLT_en_tr/HPLT.en-tr.tr",
            arrow_path
        )
    
    # 2. Filter out samples with:
    #    - short English sentences  (< 16 words)
    #    - short Turkish sentences  (< 16 words)
    #    - very long English strings (> 1500 characters)
    #    - very long Turkish strings (> 1500 characters)
    print(f"Original dataset size: {len(hplt_dataset)}")
    
    def filter_example(example):
        # English length constraints
        en_text = example["en"]
        if word_count(en_text) < 16:
            return False
        if len(en_text) > 1500:
            return False

        # Turkish length constraint
        tr_text = example.get("tr", "")
        if word_count(tr_text) < 16:
            return False
        if len(tr_text) > 1500:
            return False

        return True
    
    print(
        "Filtering dataset: keep only samples with "
        ">= 16 English words, >= 16 Turkish words, "
        "and <= 1500 English characters..."
    )
    filtered_dataset = hplt_dataset.filter(filter_example, num_proc=os.cpu_count() or 4)
    
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    # 3. Shuffle and sample 10M English words
    print("Shuffling and sampling 10M English words...")
    shuffled_dataset = filtered_dataset.shuffle(seed=42)
    
    selected_indices = []
    total_en_words = 0
    target_words = 10_000_000
    
    for i in range(len(shuffled_dataset)):
        en_text = shuffled_dataset[i]['en']
        words = word_count(en_text)
        total_en_words += words
        selected_indices.append(i)
        
        if total_en_words >= target_words:
            break
            
    print(f"Selected {len(selected_indices)} samples with a total of {total_en_words} English words.")
    
    subset = shuffled_dataset.select(selected_indices)
    
    subset_save_path = "data/arrow/HPLT_10M_subset"
    print(f"Saving 10M subset arrow dataset to {subset_save_path}...")
    subset.save_to_disk(subset_save_path)
    
    # Save the Turkish and English sentences separately as raw text files for pretraining
    txt_save_dir = "data/HPLT_10M_subset"
    os.makedirs(txt_save_dir, exist_ok=True)
    
    en_path = os.path.join(txt_save_dir, "HPLT_10M.en")
    tr_path = os.path.join(txt_save_dir, "HPLT_10M.tr")
    
    print(f"Saving separate English and Turkish text files to {txt_save_dir}...")
    with open(en_path, "w", encoding="utf-8") as f_en, open(tr_path, "w", encoding="utf-8") as f_tr:
        for item in subset:
            f_en.write(item["en"].replace("\n", " ").strip() + "\n")
            f_tr.write(item["tr"].replace("\n", " ").strip() + "\n")
    
    # 4. Analyze the sample dataset
    print("Analyzing the sample dataset...")
    analyze_dataset(subset, "HPLT_10M_subset")

if __name__ == "__main__":
    extract_10m_subset()
