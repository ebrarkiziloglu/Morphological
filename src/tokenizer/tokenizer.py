import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="multilingual_corpus.txt",
    model_prefix='spModel_40k',
    vocab_size=40000,
    pad_id=0,                
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[CLS]',
    eos_piece='[SEP]',
    user_defined_symbols='[MASK]',
    model_type='bpe'
)

from transformers import DebertaV2Tokenizer
tokenizer_deberta = DebertaV2Tokenizer(
    vocab_file  = "spModel_40k.model",
    max_len = 512,
)
print(tokenizer_deberta)
tokenizer_deberta.save_pretrained("src/tokenizer/tokenizer_40k")