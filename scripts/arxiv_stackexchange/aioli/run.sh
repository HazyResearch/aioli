#!/bin/bash


ETA=0.2
UPDATE_STEPS=250
PRIOR_STEPS=250
OHF=0.625


SLIMPAJAMA_DIR=your_directory

# Aioli-unrestricted.
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed $SEED \
        --max_steps 5000 \
        --sample_rule mixture \
        --slice_list arxiv stackexchange \
        --model EleutherAI/pythia-160m \
        --num_ckpts 20 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --aioli \
        --lp_rounds 4 \
        --lp_steps 4 \
        --eta ${ETA} \
        --ema 0.1 \
        --update_steps ${UPDATE_STEPS} \
        --one_hot_factor ${OHF} \
        --aioli_normalize_A \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention

done 

