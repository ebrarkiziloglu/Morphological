import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Regex
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# 1. Initialize an empty WordLevel tokenizer
# The '[UNK]' token will be used for any character not seen in the corpus
tk_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

# 2. Add a Pre-tokenizer that splits strings into individual characters
# The Regex(".") matches every single character independently
tk_tokenizer.pre_tokenizer = Split(Regex("."), behavior="isolated")

# 3. Define the special tokens
special_tokens = ["[BOS]", "[PAD]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# 4. Setup the trainer
# We don't set a vocab_size limit since we want all unique characters.
trainer = WordLevelTrainer(
    special_tokens=special_tokens,
    min_frequency=1, # Change this to 2+ if you want to filter out rare chars
    show_progress=True
)

# 5. Read the dataset
print("Reading corpus...")
with open("./multilingual_corpus.txt", "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

# 6. "Train" (this simply scans the corpus for all unique characters)
print("Building character vocabulary...")
tk_tokenizer.train_from_iterator(lines, trainer=trainer)
print(f"Vocabulary size: {tk_tokenizer.get_vocab_size()}")

# 7. Wrap into HF Fast Tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tk_tokenizer,
    model_max_length=512,
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

tokenizer_path = "./src/tokenizer/tokenizer_char/"
os.makedirs(tokenizer_path, exist_ok=True)

# 8. Save
tokenizer.save_pretrained(tokenizer_path)

# ---- LOAD AND TEST ----
tokenizer_loaded = AutoTokenizer.from_pretrained(tokenizer_path)

print("\n--- Testing Tokenizer ---")
test_sentences = [
    "Merhaba, bugün nasılsın?",
    "Hello, how are you doing today?",
    "Heute ist das Wetter schön.",
]

for sentence in test_sentences:
    print(f"Original:  {sentence}")
    print(f"Pieces:    {tokenizer_loaded.tokenize(sentence)}")
    # The IDs will correspond to individual character mappings
    print(f"Token IDs: {tokenizer_loaded.encode(sentence)}")
    print(f"Decoded:   {tokenizer_loaded.decode(tokenizer_loaded.encode(sentence))}")
    print("-" * 40)
