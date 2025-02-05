#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12000:00
python train.py \
  --ques_file ../../../Bert_datas/Full_turn_query/train_test_dataset_bert_full_turn_poop.tsv \
  --doc_file ../../../Bert_datas/documents.tsv \
  --qrels ../../../Bert_datas/qrels \
  --train_pairs ../../../Bert_datas/Full_turn_query/train_full_turn.pairs \
  --valid_run ../../../Bert_datas/Full_turn_query/test_full_turn.run\
  --model_out_dir models/bert_fullturn \
  --img_embed_dict ../../../Bert_datas/img_feature_all.pkl \
  --img_tag_dict ../../../Bert_datas/Full_turn_query/train_test_dataset_bert_full_turn_poop.json \


