# Aioli: A unified optimization framework for language model data mixing


<div align="center" >
    <img src="assets/aioli_logo_final.svg" height=350 alt="Aioli logo" style="margin-bottom:px"/> 
</div>
<br/>

## Overview

This repository contains replication code for the following paper:

> Aioli: A unified optimization framework for language model data mixing \
> Mayee F. Chen*, Michael Y. Hu*, Nicholas Lourie, Kyunghyun Cho, Christopher Ré \
> [arxiv](https://arxiv.org/abs/2411.05735)

Data mixing is an important step in data curation where practitioners must identify the optimal mixture of data groups to train on (i.e., code, law, math).
While a brute-force search over the mixture proportions is a common technique in practice, this approach requires many training runs. Recent alternatives propose to algorithmically learn mixture proportions more efficiently; however, they are not well-understood and can sometimes even underperform a simple stratified sampling baseline. 

In our work, we find that many mixing methods can be written as an optimization problem subject to a linear mixing law that describes the assumed relationship between loss and mixture proportion. 
We find that while methods are accurate in the parameterization of their mixing laws, they are incorrect in how they estimate the mixing law parameters. Notably, these inaccuracies in their parameters are correlated with when these methods do worse than stratified sampling, providing a more complete explanation of when and why data mixing methods fail.

Given these insights, the Aioli algorithm estimates the mixing law parameters from the current training trajectory and dynamically updates the mixture proportions. Aioli is able to consistently outperform stratified sampling and can enhance existing mixing methods by adjusting their learned proportions throughout the full training run.

## Dependencies

To install dependencies:
```bash
git clone https://github.com/HazyResearch/aioli.git
cd aioli/
pip install -r requirements.txt
```

## Usage

To run Aioli, refer to sample scripts `scripts/arxiv_stackexchange/aioli/run.sh`, `scripts/arxiv_books_stackexchange/aioli/run.sh`, and `scripts/full/aioli/run.sh`:

```bash
SLIMPAJAMA_DIR=your_directory
python3 main.py \
      --task_name slimpj \
      --train_data_dir $SLIMPAJAMA_DIR \
      --val_data_dir $SLIMPAJAMA_DIR \
      --selection_seed 0 \
      --max_steps 5000 \
      --sample_rule mixture \
      --slice_list arxiv stackexchange \
      --model EleutherAI/pythia-160m \
      --num_ckpts 20 \
      --batch_size 8 \
      --context_length 2048 \
      --lr 0.0005 \
      --aioli \
      --lp_rounds 4 \
      --lp_steps 4 \
      --eta 0.2 \
      --ema 0.1 \
      --update_steps 250 \
      --one_hot_factor 0.625 \
      --aioli_normalize_A \
      --checkpoint 0 \
      --lr_scheduler linear_warmup_cosine \
      --warmup_steps 500 \
      --use_flash_attention
```

Aioli parameters:
- `--aioli`: always set to use the algorithm.
- `--lp_rounds`: the number of sweeps for each mixture in LearnParams subroutine (Algorithm 2 in paper). Higher=more accurate $A^t$, but more of training is then allocated to learning $A^t$. Recommended to try 4 or 6.
- `--lp_steps`: the number of steps to train each mixture in LearnParams. Recommended to set to 4.
- `--eta`: step size in exponential gradient descent update. Recommended to try values like 0.1, 0.2, 0.5.
- `--ema`: exponential moving average parameter (optional, lower=more recency bias). Recommended to not set this at first.
- `--update_steps`: how often to re-estimate $A^t$. Determines the number of rounds T = max_steps / update_steps.
- `--one_hot_factor`: how much to smooth one-hot distributions used in learning $A^t$ (1 = no smoothing). That is, your training batch will contain batch_size * one_hot_factor samples of one group, and equal samples for the other groups. Recommended to set this to be less than 1.
- `--aioli_normalize_A`: always set to normalize $A^t$.


In the above command, Aioli is run in the "unrestricted" setting, which corresponds to running Aioli without any prior knowledge, in which the mixing law parameters and mixture proportions are updated starting from the beginning of training.
Another way that Aioli can be used is in the "restricted" setting: after a model is already trained for `--prior_steps` using weights `--aioli_prior` (i.e., learned from some other data mixing method), Aioli can dynamically adjust this prior. 
This is an effective way to enhance any existing data mixing method, especially when the existing method's weights are potentially noisy (e.g., learned over shortened runs).


## Reproducing results in the paper 

We have provided sample scripts for several datasets (Arxiv/Stackexchange, Arxiv/BookStackexchange, full SlimPajama, and instruction-tuning tasks).
For all the sample scripts, replace `SLIMPAJAMA_DIR` with where you want to keep the SlimPajama dataset. For the instruction-tuning experiments, replace `NI_DIR` with the path of the [natural instructions repository](https://github.com/allenai/natural-instructions).


### Running baselines

We've also implemented several baselines within this repo. You can run baselines by doing the following:
- Grid search: run `scripts/full/sweep/run.sh` or equivalent in other directories.
- Data Mixing Laws:
      - run `scripts/full/sweep/run.sh` or equivalent in other directories
      - derive the optimal mix by running `python mixing_law_fitting.py linreg --run_dir /path/to/results --proportions_file /path/to/proportions/swept`
      - run `scripts/full/sweep/run_final.sh`, filling in your optimal mix
- Skill-it: run `scripts/full/skillit/graph_learning.sh`. The skills graph needs to be created using `python mixing_law_fitting.py skills_graph --run_dir /path/to/results --seed seed`, then `run.sh`
- DoGE: run `scripts/full/doge/run_proxy.sh`. Get the weights by loading `outputs/.../seed_{seed}_doge_avg_dw.pkl`, then `run_final.sh`
- DoReMi: run `scripts/full/doremi/run_reference.sh`, then `run_proxy.sh`. Get the weights by loading `outputs/.../seed_{seed}_doremi_avg_dw.pkl`, then `run_final.sh`

Similar baseline scripts are available for Arxiv/Stackexchange and Arxiv/Book/Stackexchange in `scripts/{arxiv_stackexchange, arxiv_books_stackexchange}/`. 

### Mixing law analysis 

To reproduce our analyses for mixing laws, see the README in the `analysis` directory.

## Citation

```
@misc{chen2024aioliunifiedoptimizationframework,
      title={Aioli: A Unified Optimization Framework for Language Model Data Mixing}, 
      author={Mayee F. Chen and Michael Y. Hu and Nicholas Lourie and Kyunghyun Cho and Christopher Ré},
      year={2024},
      eprint={2411.05735},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.05735}, 
}
```

## Contact

Feel free to create an issue or email Mayee Chen (<mfchen@stanford.edu>) and Michael Hu (<michael.hu@nyu.edu>).
