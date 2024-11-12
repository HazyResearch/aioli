# Mixing law analysis

Notebooks that are of the format `{k}_{static/dynamic}_law.ipynb` fit the log-linear static and linear dynamic mixing laws over training sweeps for $m = 2, 3$. `instruction_{static/dynamic}_law.ipynb` contain our results on the mixing law parameterization for instruction-tuning tasks. 

Notebooks that are of the format `{k}_{method}_parameters.ipynb` compute each method's $A^t$ and $A^{t \star}$ (conditioned on that method's trajectory) for $m = 2, 3$ and for methods = Skill-It, DoGE, DoReMi, Aioli.

All fitted parameters are saved to `law_results/`. 

Notebooks that are of the format `{k}_measurements.ipynb` load in the fitted parameters and compute $\text{sim}(\tilde{A}^t, A^{t \star})$ for each method. 

To use these notebooks, you need to launch static and dynamic training sweeps first, such as in `scripts/arxiv_stackexchange/sweep/`. 

Lastly, we thank the authors of [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952). The code for fitting the static mixing law was originally adopted from their codebase, located [here](https://github.com/yegcjs/mixinglaws).

#### Static mixing laws

We study how well the mixing law $L_i(p) = c_i + b_i \exp(\sum_{j = 1}^m -A_{ij} p_j)$ holds. To conduct a training sweep over static mixture proportions, see `scripts/{arxiv_stackexchange, arxiv_books_stackexchange, instruction}/sweep/run.sh`.

Once you have conducted the training sweep, please refer to `analysis/2_static_law.ipynb` and `analysis/3_static_law.ipynb` for the fitting code for $m=2, 3$ groups.

#### Dynamic mixing laws

We study how well the mixing law $L_i^{t+1}(p) = L_i^t(p) - \sum_{j = 1}^m A_{ij}^t p_j^t$ holds. To conduct a training sweep over dynamic mixture proportions, see `scripts/{arxiv_stackexchange, arxiv_books_stackexchange, instruction}/sweep/dynamic/run_child.sh`. In each of these, we load a checkpoint of a partially trained model (i.e., `run_parent.sh`) and sweep over a small amount of training steps. 

Once you have conducted the training sweep, please refer to `analysis/2_dynamic_law.ipynb` and `analysis/3_dynamic_law.ipynb` for the fitting code for $m=2, 3$ groups.

#### Analyzing mixing law parameters

For each dynamic mixing method (Aioli, DoGE, DoReMi, Skill-It), we truncate the training run and then sweep over a small amount of training steps. 
Once you have conducted the training sweep, please refer to `analysis/{2, 3}_{aioli, doge, doremi, skillit}_parameters.ipynb` to obtain each method's $A^t$ and $A^{t \star}$, and refer to `analysis/{2, 3}_measurements.ipynb` to compute $\text{sim}(\tilde{A}^t, A^{t \star})$.