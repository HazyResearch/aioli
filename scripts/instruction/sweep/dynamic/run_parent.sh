#!/bin/bash

NI_DIR=your_directory
# to easily generate the rest of the sweep scripts, run run_instruction_sweep.py.
python3 main.py \
    --task_name instruction \
    --train_data_dir $NI_DIR \
    --val_data_dir $NI_DIR \
    --selection_seed 0 \
    --max_steps 1000 \
    --sample_rule mixture \
    --proportions 1 1 1 1 1 1 1 1 1 \
    --model EleutherAI/pythia-160m \
    --num_ckpts 10 \
    --batch_size 8 \
    --context_length 2048 \
    --lr 0.00001 \
    --lr_scheduler linear \
    --warmup_steps 100 \
    --break_steps 500

