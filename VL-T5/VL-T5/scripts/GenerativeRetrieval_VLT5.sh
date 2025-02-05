# The name of experiment
name=VLT5

output=snap/GenerativeRetrieval/$name

PYTHONPATH=$PYTHONPATH:./src \
python src/generative_retrieval.py \
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
        --output $output ${@:2} \
        --num_beams 10 \
        --max_text_length 1024 \
        --batch_size 4 \
        --valid_batch_size 64 \
        --dump_path snap/result_vlt5.pkl \
        --load snap/pretrain/VLT5/Epoch30 \

