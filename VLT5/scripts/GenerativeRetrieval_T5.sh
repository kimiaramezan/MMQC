# The name of experiment
name=T5

output=snap/GenerativeRetrieval/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/generative_retrieval_t5.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --local-rank 0\
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 15 \
        --max_text_length 50 \
        --batch_size 16 \
        --valid_batch_size 16 \
        --dump_path snap/result_vlt5.pkl \
         # --load snap/pretrain/VLT5/Epoch30 \
