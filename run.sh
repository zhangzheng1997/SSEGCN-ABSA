#!/bin/bash
# * laptop

# * SSEGCN
python ./train.py --model_name ssegcn --dataset laptop --seed 1000 --num_epoch 40 --vocab_dir ./dataset/Laptops_corenlp --cuda 0  

# * SSEGCN with Bert
# python ./train.py --model_name ssegcn_bert --dataset laptop --seed 1000 --bert_lr 2e-5 --num_epoch 10 --hidden_dim 768 --max_length 100 --cuda 0  


# * restaurant

# * SSEGCN
# python ./train.py --model_name ssegcn --dataset restaurant --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Restaurants_corenlp --cuda 0  
# * SSEGCN with Bert
# python ./train.py --model_name ssegcn_bert --dataset restaurant --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 

# * twitter

# * SSEGCN
# python ./train.py --model_name ssegcn --dataset twitter --seed 1000 --num_epoch 40 --vocab_dir ./dataset/Tweets_corenlp --cuda 0 

# * SSEGCN with Bert
# python ./train.py --model_name ssegcn_bert --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 
