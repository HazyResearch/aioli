#!/bin/bash

SLIMPAJAMA_DIR=your_directory

# skill-it baseline in unrestricted setting.
for SEED in 0 1 2 3 4
do 
    for ETA in 0.2
    do 
        for UPDATE_STEPS in 500
        do 
        python3 main.py \
            --task_name slimpj \
            --train_data_dir $SLIMPAJAMA_DIR \
            --val_data_dir $SLIMPAJAMA_DIR \
            --selection_seed ${SEED} \
            --max_steps 5000 \
            --sample_rule mixture \
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
            --skillit \
            --update_steps $UPDATE_STEPS \
            --skillit_window 3 \
            --eta $ETA \
            --graph_path ./skillit_graphs/slimpj_arxiv_stackexchange_normalized_seed_${SEED}.npy 

        done 
    done
done 