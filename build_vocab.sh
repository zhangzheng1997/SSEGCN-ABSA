#!/bin/bash
# build vocab for different datasets

python ./prepare_vocab.py --data_dir dataset/Restaurants_corenlp --vocab_dir dataset/Restaurants_corenlp
python ./prepare_vocab.py --data_dir dataset/Laptops_corenlp --vocab_dir dataset/Laptops_corenlp
python ./prepare_vocab.py --data_dir dataset/Tweets_corenlp --vocab_dir dataset/Tweets_corenlp
