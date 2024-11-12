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
    --proportions 0.014379 0.009831 0.129966 0.277409 0.143394 0.102429 0.035138 0.004082 0.283373 \
    --model EleutherAI/pythia-160m \
    --num_ckpts 10 \
    --batch_size 8 \
    --context_length 2048 \
    --lr 0.00001 \
    --lr_scheduler linear \
    --warmup_steps 100 \
    --resume_model_path ./saved_model_optim/instruction_pythia-160m_1000_break_500_mixture_weights_0.0143790.0098310.1299660.2774090.1433940.1024290.0351380.0040820.283373_static_lr_1e-05_seed_0.pt \
    --remaining_steps 100 
