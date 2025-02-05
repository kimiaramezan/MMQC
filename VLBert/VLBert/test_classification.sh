#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12000:00
python test_classification.py \
  --doc_file ../../mqc/data/facet/cwdocs.tsv \
  --qrels ../../mqc/data/facet/qrels \
  --run ../../mqc/data/facet/train.qrel \
  --model_weights models/bert/weights.p \
  --img_embed_dict /ivi/ilps/personal/yyuan/img_feature_all.pkl \
  --out_path models/bert/train_facets.pkl \