#!/bin/bash
SEED=0
N_GRID=10
SLIMPAJAMA_DIR=your_directory

# Loads the model at --resume_model_path and sweeps over mixtures for the next --remaining_steps.
# to easily generate the rest of the sweep scripts, run run_slimpj_sweep.py.
for P1 in 0 1 2 3 4 5 6 7 8 9 10
do 
    P2=$(expr $N_GRID - $P1)
    python3 main.py \
        --task_name slimpj \
        --train_data_dir $SLIMPAJAMA_DIR \
        --val_data_dir $SLIMPAJAMA_DIR \
        --selection_seed ${SEED} \
        --max_steps 5000 \
        --sample_rule mixture \
        --proportions 0 10 \
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
        --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_4000_mixture_arxiv_stackexchange_weights_${P1}${P2}_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
        --remaining_steps 100
done 
