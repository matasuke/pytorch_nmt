# preprocess small parallel corpora
python preprocessor/text_preprocessor.py \
    -data_path data/corpora/small_parallel.ja \
    -save_path data/preprocessor/small_parallel.8000.ja.pkl \
    -max_vocab_size 8000

python preprocessor/text_preprocessor.py \
    -data_path data/corpora/small_parallel.en \
    -save_path data/preprocessor/small_parallel.8000.en.pkl \
    -max_vocab_size 8000
