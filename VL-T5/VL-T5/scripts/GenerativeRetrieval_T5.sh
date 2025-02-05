# The name of experiment
name=T5

output=snap/GenerativeRetrieval/$name

PYTHONPATH=$PYTHONPATH:./src \
    python src/generative_retrieval_t5.py \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --gen_max_length 50 \
        --backbone 't5-base' \
        --use_vision False \
        --output $output ${@:2} \
        --num_beams 10 \
        --max_text_length 1024 \
        --batch_size 4 \
        --valid_batch_size 64 \
        --dump_path snap/result_t5.pkl \
        # --load snap/pretrain/VLT5/Epoch30 \
