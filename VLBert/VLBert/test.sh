#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12000:00
python test.py \
  --ques_file  ../../../New_datas_LLaVA/train_test_visbert_poop.tsv \
  --doc_file ../../../Bert_datas/documents.tsv  \
  --qrels  ../../../Bert_datas/qrels  \
  --run ../../../Bert_datas/test.run \
  --model_weights models/ours/weights.p \
  --img_embed_dict ../../../Bert_datas/img_feature_all.pkl\
  --out_path models/ours/test.pkl \
  --img_tag_dict ../../../Bert_datas/train_test_visbert_poop.json \


