#!/bin/bash

SEED=0
BREAK=2000
SLIMPAJAMA_DIR=your_directory

# Trains the model for $BREAK number of steps and saves the model.
# to easily generate the rest of the sweep scripts, run run_slimpj_sweep.py.
python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 5 5 \
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
    --break_steps $BREAK
