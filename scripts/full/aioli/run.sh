

# AIOLI and its ablations


SLIMPAJAMA_DIR=your_directory

for SEED in 0 1 2  
do
    for eta in 1 
    do
        for ema in 0.1
        do
            # normal aioli
            python main.py \
            --task_name slimpj \
            --train_data_dir $SLIMPAJAMA_DIR \
            --val_data_dir $SLIMPAJAMA_DIR \
            --selection_seed $SEED \
            --max_steps 40000 \
            --sample_rule mixture \
            --slice_list arxiv book c4 cc github stackexchange wikipedia \
            --model EleutherAI/pythia-160m \
            --num_ckpts 20 \
            --batch_size 8 \
            --eval_batch_size 16 \
            --context_length 2048 \
            --lr 0.0005 \
            --aioli \
            --lp_rounds 2 \
            --lp_steps 10 \
            --eta $eta \
            --update_steps 200 \
            --aioli_prior 1 1 1 1 1 1 1 \
            --prior_steps 0 \
            --one_hot_factor 0.75 \
            --ema $ema \
            --aioli_normalize_A \
            --checkpoint 0 \
            --lr_scheduler linear_warmup_cosine \
            --warmup_steps 500 \
            --use_flash_attention \
            --save_model \

            # aioli static. the one change is we play the 1st A matrix for all steps --> update_steps == max_steps
            python main.py \
            --task_name slimpj \
            --train_data_dir $SLIMPAJAMA_DIR \
            --val_data_dir $SLIMPAJAMA_DIR \
            --selection_seed $SEED \
            --max_steps 40000 \
            --sample_rule mixture \
            --slice_list arxiv book c4 cc github stackexchange wikipedia \
            --model EleutherAI/pythia-160m \
            --num_ckpts 20 \
            --batch_size 8 \
            --eval_batch_size 16 \
            --context_length 2048 \
            --lr 0.0005 \
            --aioli \
            --lp_rounds 2 \
            --lp_steps 10 \
            --eta $eta \
            --update_steps 40000 \
            --aioli_prior 1 1 1 1 1 1 1 \
            --prior_steps 0 \
            --one_hot_factor 0.75 \
            --ema $ema \
            --aioli_normalize_A \
            --checkpoint 0 \
            --lr_scheduler linear_warmup_cosine \
            --warmup_steps 500 \
            --use_flash_attention \
            --save_model \

            # aioli with diagonal A
            python main.py \
            --task_name slimpj \
            --train_data_dir $SLIMPAJAMA_DIR \
            --val_data_dir $SLIMPAJAMA_DIR \
            --selection_seed $SEED \
            --max_steps 40000 \
            --sample_rule mixture \
            --slice_list arxiv book c4 cc github stackexchange wikipedia \
            --model EleutherAI/pythia-160m \
            --num_ckpts 20 \
            --batch_size 8 \
            --eval_batch_size 16 \
            --context_length 2048 \
            --lr 0.0005 \
            --aioli \
            --lp_rounds 2 \
            --lp_steps 10 \
            --bandit_diagonal \
            --eta $eta \
            --update_steps 2000 \
            --aioli_prior 1 1 1 1 1 1 1 \
            --prior_steps 0 \
            --one_hot_factor 1 \
            --ema $ema \
            --aioli_normalize_A \
            --checkpoint 0 \
            --lr_scheduler linear_warmup_cosine \
            --warmup_steps 500 \
            --use_flash_attention \
            --save_model \
        done
    done
done


