#!/bin/bash

proportions_file=$1
SEED=$2

SLIMPAJAMA_DIR=your_directory
# this script assumes you input $1 a txt file containing the proportions you want to use
# and $2 the random seed for all these proportions

while IFS= read -r line; do
    line=$(echo "$line" | tr ',' ' ') # Replace commas with spaces
    python main.py \
        --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps 40000 \
        --sample_rule mixture \
        --proportions $line \
        --slice_list arxiv book c4 cc github stackexchange wikipedia \
        --model EleutherAI/pythia-160m \
        --num_ckpts 1 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention \
        --save_model \
done < "$proportions_file
