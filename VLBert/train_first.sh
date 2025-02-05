#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12000:00

python train_first.py \
  --ques_file ../Bert_datas/Final/Full_turn_query/train_test_dataset_bert_full_turn_poop.tsv \
  --doc_file ../Bert_datas/documents.tsv \
  --qrels ../qrels \
  --train_pairs ../Bert_datas/Final/Full_turn_query/train_full_turn.pairs \
  --valid_run ../Bert_datas/Final/Full_turn_query/test_full_turn_reduced.run\
  --model_out_dir models/bert_singleturn \
  --img_embed_dict ../VL-T5/img_feature_all.pkl \
  --img_tag_dict ../Bert_datas/Final/Full_turn_query/train_test_dataset_bert_full_turn_poop.json \

# python train_first.py \
#   --ques_file ../Bert_datas/Full_turn_negative/train_test_dataset_bert_full_turn_poop.tsv \
#   --doc_file ../Bert_datas/documents.tsv \
#   --qrels ../qrels \
#   --train_pairs ../Bert_datas/Full_turn_negative/train_full_turn.pairs \
#   --valid_run ../Bert_datas/Full_turn_negative/test_full_turn_reduced.run\
#   --model_out_dir models/bert_fullturn \
#   --img_embed_dict ../VL-T5/img_feature_all.pkl \
#   --img_tag_dict ../Bert_datas/Full_turn_negative/train_test_dataset_bert_full_turn_poop.json \

