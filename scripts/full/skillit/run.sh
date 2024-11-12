# skill-it

SLIMPAJAMA_DIR=your_directory
max_steps=40000

for SEED in 0 1 2 
do 
    for ETA in 0.2 
    do 
        for UPDATE_STEPS in 400
        do 
        python main.py --task_name slimpj \
            --train_data_dir $SLIMPAJAMA_DIR \
            --val_data_dir $SLIMPAJAMA_DIR \
            --selection_seed ${SEED} \
            --max_steps 40000 \
            --sample_rule mixture \
            --slice_list arxiv book c4 cc github stackexchange wikipedia \
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
            --graph_path ./output/skills_graphs/skills_graph_${SEED}.npy \
            --save_model 
            
        done 
    done