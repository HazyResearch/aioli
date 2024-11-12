#!/bin/bash

SLIMPAJAMA_DIR=your_directory
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps 10000 \
        --sample_rule stratified \
        --slice_list arxiv book stackexchange \
        --model EleutherAI/pythia-160m \
        --num_ckpts 40 \
        --batch_size 4 \
        --context_length 2048 \
        --lr 0.0005 \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention \
        --gradient_accumulation_steps 2 \
        --doremi \
        --doremi_mu 0.01 \
        --doremi_reference_model_path ./saved_models/slimpj_pythia-160m_from_scratch_5000_stratified_arxiv_book_stackexchange_static_lr_0.0005_linear_warmup_cosine_seed_${SEED}.pt

done 