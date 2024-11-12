SLIMPAJAMA_DIR=your_directory

for SEED in 0 1 2 
do 
    python main.py --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps 40000 \
        --break_steps 40000 \
        --sample_rule mixture \
        --proportions [FINAL_PROPORTIONS_HERE] \
        --slice_list arxiv book c4 cc github stackexchange wikipedia \
        --model EleutherAI/pythia-160m \
        --num_ckpts 5 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention \
        --save_model
        
done 