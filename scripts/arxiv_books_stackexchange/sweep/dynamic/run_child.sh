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
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.0845890.3106480.604763_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100

python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.0973330.7524720.150196_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100


python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.1489040.4998770.351219_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100


python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.1653160.0925740.74211_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100


python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.2909910.3057860.403223_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100


python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.4392260.0944840.466289_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100


python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.5006440.3987250.100631_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100


python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.6576670.106660.235673_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100


python3 main.py \
    --task_name slimpj \
    --train_data_dir $SLIMPAJAMA_DIR \
    --val_data_dir $SLIMPAJAMA_DIR \
    --selection_seed ${SEED} \
    --max_steps 5000 \
    --sample_rule mixture \
    --proportions 0 0 1 \
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
    --resume_model_path ./saved_model_optim/slimpj_pythia-160m_from_scratch_5000_break_2000_mixture_arxiv_book_stackexchange_weights_0.8955440.0611920.043264_static_lr_0.0005_linear_warmup_cosine_seed_0.pt \
    --remaining_steps 100

