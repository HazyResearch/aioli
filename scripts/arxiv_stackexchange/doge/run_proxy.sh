#!/bin/bash
SEED=0

SLIMPAJAMA_DIR=your_directory

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps 5000 \
        --sample_rule stratified \
        --slice_list arxiv stackexchange \
        --model EleutherAI/pythia-160m \
        --num_ckpts 20 \
        --batch_size 6 \
        --doge_val_batch_size 2 \
        --eval_batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention \
        --doge \
        --doge_mu 0.01

done 