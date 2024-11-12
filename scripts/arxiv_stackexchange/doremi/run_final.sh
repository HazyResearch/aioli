#!/bin/bash

SLIMPAJAMA_DIR=your_directory

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed 0 \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0.3702907860279083 0.6297092139720917 \
    --slice_list arxiv stackexchange \
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
    --proportions 0.37893620133399963 0.6210637986660004 \
    --slice_list arxiv stackexchange \
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
    --proportions 0.3511049151420593 0.6488950848579407 \
    --slice_list arxiv stackexchange \
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
    --proportions 0.3649815618991852 0.6350184381008148 \
    --slice_list arxiv stackexchange \
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
    --proportions 0.36395448446273804 0.636045515537262 \
    --slice_list arxiv stackexchange \
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
