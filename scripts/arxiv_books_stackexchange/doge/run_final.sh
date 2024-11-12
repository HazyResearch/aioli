#!/bin/bash

SLIMPAJAMA_DIR=your_directory

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 0 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.59173548 0.13224243 0.27602303 \
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
    --custom_name doge_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 1 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.62134719 0.11532071 0.26333308 \
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
    --custom_name doge_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 2 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.59243822 0.11080202 0.29675922 \
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
    --custom_name doge_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 3 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.63508421 0.11062542 0.25429085 \
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
    --custom_name doge_final

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 4 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.58928579 0.13616571 0.27454644 \
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
    --custom_name doge_final

