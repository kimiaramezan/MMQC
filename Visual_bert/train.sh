#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12000:00
python train.py \
  --ques_file ../../../New_datas_LLaVA/train_test_visbert_poop.tsv \
  --doc_file ../../../Bert_datas/documents.tsv \
  --qrels ../../../Bert_datas/qrels \
  --train_pairs ../../../Bert_datas/train.pairs \
  --valid_run ../../../Bert_datas/test.run\
  --model_out_dir models/bert_1 \
  --img_embed_dict ../../../Bert_datas/img_feature_all.pkl \
  --img_tag_dict ../../../Bert_datas/train_test_visbert_poop.json \


