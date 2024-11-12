#!/bin/bash


SLIMPAJAMA_DIR=your_directory

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 0 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.3181622 0.17973898 0.50209862 \
    --slice_list arxiv book stackexchange \
    --model EleutherAI/pythia-160m \
    --num_ckpts 20 \
    --batch_size 8 \
    --context_length 2048 \
    --lr 0.0005 \
    --checkpoint 0 \
    --lr_scheduler linear_warmup_cosine \
    --warmup_steps 500 \
    --use_flash_attention \
    --custom_name doremi_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 1 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.30559674 0.19329953 0.50110328 \
    --slice_list arxiv book stackexchange \
    --model EleutherAI/pythia-160m \
    --num_ckpts 20 \
    --batch_size 8 \
    --context_length 2048 \
    --lr 0.0005 \
    --checkpoint 0 \
    --lr_scheduler linear_warmup_cosine \
    --warmup_steps 500 \
    --use_flash_attention \
    --custom_name doremi_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 2 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.2915031 0.17897391 0.52952266 \
    --slice_list arxiv book stackexchange \
    --model EleutherAI/pythia-160m \
    --num_ckpts 20 \
    --batch_size 8 \
    --context_length 2048 \
    --lr 0.0005 \
    --checkpoint 0 \
    --lr_scheduler linear_warmup_cosine \
    --warmup_steps 500 \
    --use_flash_attention \
    --custom_name doremi_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 3 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.30619839 0.18335673 0.51044524 \
    --slice_list arxiv book stackexchange \
    --model EleutherAI/pythia-160m \
    --num_ckpts 20 \
    --batch_size 8 \
    --context_length 2048 \
    --lr 0.0005 \
    --checkpoint 0 \
    --lr_scheduler linear_warmup_cosine \
    --warmup_steps 500 \
    --use_flash_attention \
    --custom_name doremi_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 4 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.32210311 0.15809093 0.51980549 \
    --slice_list arxiv book stackexchange \
    --model EleutherAI/pythia-160m \
    --num_ckpts 20 \
    --batch_size 8 \
    --context_length 2048 \
    --lr 0.0005 \
    --checkpoint 0 \
    --lr_scheduler linear_warmup_cosine \
    --warmup_steps 500 \
    --use_flash_attention \
    --custom_name doremi_final

