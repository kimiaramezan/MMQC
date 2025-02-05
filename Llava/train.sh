#!/bin/bash


python src/train_textonly.py \
    --num_beams 10 \
    --temperature 1\
    --do_sample False\
    --max_new_tokens 30 \
    --train_dataset "./First_turn_img/train_dataset_LLaVA_first_turn_keywords.jsonl" \
    --test_dataset "./First_turn_img/test_dataset_LLaVA_first_turn_keywords.jsonl" \
    --ground_truth './First_turn_img/test_dataset_LLaVA_first_turn_keywords.jsonl' \
    --keywords_file './filtered_keywords.tsv' \
    --text_only False \
    --batch_size_train 4 \
    --batch_size_test 1 \
    --lora_rank 16\
    --max_length 256 \
    --lr 1e-5 \
    --model_save_dir '/workspace/models/poop'

    # --train_dataset "./Full_turn_text/train_dataset_LLaVA_full_turn_keywords.jsonl" \
    # --test_dataset "./Full_turn_text/test_dataset_LLaVA_full_turn_keywords.jsonl" \
    # --ground_truth './Full_turn_text/test_dataset_LLaVA_full_turn_keywords.jsonl' \