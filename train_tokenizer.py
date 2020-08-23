import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input='./corpus_train.txt',
    model_prefix='tokenizer_65536',
    vocab_size=65536,
    character_coverage=0.9995,
    user_defined_symbols=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    input_sentence_size=1000000,
    shuffle_input_sentence=True
)
