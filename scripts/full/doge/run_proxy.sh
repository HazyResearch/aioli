#!/bin/bash

SLIMPAJAMA_DIR=your_directory

# DOGE
for SEED in 0 1 2 
do 
    for max_steps in 40000 
    do
        for mu in 0.1
        do
            python main.py \
            --task_name slimpj \
            --train_data_dir $SLIMPAJAMA_DIR \
            --val_data_dir $SLIMPAJAMA_DIR \
            --selection_seed ${SEED} \
            --max_steps $max_steps \
            --sample_rule stratified \
            --slice_list arxiv book c4 cc github stackexchange wikipedia \
            --model EleutherAI/pythia-160m \
            --num_ckpts 4 \
            --batch_size 4 \
            --doge_val_batch_size 4 \
            --eval_batch_size 8 \
            --context_length 2048 \
            --lr 0.0005 \
            --checkpoint 0 \
            --lr_scheduler linear_warmup_cosine \
            --warmup_steps 500 \
            --use_flash_attention \
            --doge \
            --doge_mu $mu \
        done
    done
done 