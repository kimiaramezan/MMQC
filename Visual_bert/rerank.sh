#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12000:00
python rerank.py \
  --ques_file ../../mqc/data/facet/facets.tsv \
  --doc_file ../../mqc/data/facet/cwdocs.tsv \
  --run ../../mqc/data/mydata/test.qrel1 \
  --model_weights models/ours/weights.p \
  --img_embed_dict /ivi/ilps/personal/yyuan/img_feature_all.pkl \
  --out_path models/ours/test_img_1.run \
  --img_tag_dict /home/yyuan/MQC/VL-T5/datasets/all.json \
