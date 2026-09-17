from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import os

with open("./multilingual_corpus.txt", "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

# special tokens
special_tokens = ["[BOS]", "[PAD]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# trainind
tk_tokenizer = SentencePieceBPETokenizer(unk_token="[UNK]")
tk_tokenizer.train_from_iterator(
    lines,
    vocab_size=40000,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens
)
print(tk_tokenizer)

# Convert to HF tokenizer
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

tokenizer_path = "./src/tokenizer/tokenizer_40k/"

if not os.path.exists(tokenizer_path):
    os.makedirs(tokenizer_path)

# Save
tokenizer.save_pretrained(tokenizer_path)

# Load test
tokenizer_loaded = AutoTokenizer.from_pretrained(tokenizer_path)

print(tokenizer_loaded)

# Test
test_sentences = [
    "Merhaba, bugün nasılsın?",
    "Nasılsın? Bugün hava çok güzel.",
    "Yarın ne yapacaksın?",
    "Hello, how are you doing today?",
    "What are your plans for tomorrow?",
    "The weather is nice today.",
    "Hallo, wie geht es dir heute?",
    "Was machst du morgen?",
    "Heute ist das Wetter schön.",
]

for sentence in test_sentences:
    print(f"Original: {sentence}")
    print(f"Pieces:   {tokenizer_loaded.tokenize(sentence)}")
    print(f"Token IDs: {tokenizer_loaded.encode(sentence)}")
    print(f"Decoded:  {tokenizer_loaded.decode(tokenizer_loaded.encode(sentence))}")
    print()