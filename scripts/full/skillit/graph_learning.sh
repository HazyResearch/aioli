SLIMPAJAMA_DIR=your_directory
max_steps=40000

for SEED in 0 1 2 
do 
    python main.py --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps $max_steps \
        --sample_rule mixture \
        --proportions 1 0 0 0 0 0 0  \
        --slice_list arxiv book c4 cc github stackexchange wikipedia\
        --model EleutherAI/pythia-160m \
        --num_ckpts 2 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention
    
    python main.py --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps $max_steps \
        --sample_rule mixture \
        --proportions 0 1 0 0 0 0 0  \
        --slice_list arxiv book c4 cc github stackexchange wikipedia\
        --model EleutherAI/pythia-160m \
        --num_ckpts 2 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention
    
    python main.py --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps $max_steps \
        --sample_rule mixture \
        --proportions 0 0 1 0 0 0 0  \
        --slice_list arxiv book c4 cc github stackexchange wikipedia\
        --model EleutherAI/pythia-160m \
        --num_ckpts 2 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --filter_val_skills \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention
    
    python main.py --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps $max_steps \
        --sample_rule mixture \
        --proportions 0 0 0 1 0 0 0  \
        --slice_list arxiv book c4 cc github stackexchange wikipedia\
        --model EleutherAI/pythia-160m \
        --num_ckpts 2 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --filter_val_skills \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention 

    python main.py --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps $max_steps \
        --sample_rule mixture \
        --proportions 0 0 0 0 1 0 0  \
        --slice_list arxiv book c4 cc github stackexchange wikipedia\
        --model EleutherAI/pythia-160m \
        --num_ckpts 2 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --filter_val_skills \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention

    python main.py --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps $max_steps \
        --sample_rule mixture \
        --proportions 0 0 0 0 0 1 0  \
        --slice_list arxiv book c4 cc github stackexchange wikipedia\
        --model EleutherAI/pythia-160m \
        --num_ckpts 2 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --filter_val_skills \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention

    python main.py--task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps $max_steps \
        --sample_rule mixture \
        --proportions 0 0 0 0 0 0 1  \
        --slice_list arxiv book c4 cc github stackexchange wikipedia\
        --model EleutherAI/pythia-160m \
        --num_ckpts 2 \
        --batch_size 8 \
        --context_length 2048 \
        --lr 0.0005 \
        --filter_val_skills \
        --checkpoint 0 \
        --lr_scheduler linear_warmup_cosine \
        --warmup_steps 500 \
        --use_flash_attention 
done