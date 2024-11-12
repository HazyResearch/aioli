#!/bin/bash
SEED=0
SLIMPAJAMA_DIR=your_directory

# to easily generate the rest of the sweep scripts, run run_3_slimpj_sweep.py.

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 1 0 0 \
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
    --break_steps 2000