import os
import pickle
from transformers import DataCollatorForLanguageModeling
import wandb 
import pandas as pd 
import torch 
import numpy as np


def parse_resumed_model_cpkt(args):
    path = args.resume_model_path

    if "weightschedule" in path:
        weight_str = path.split("weightschedule_")[-1]
    else:
        weight_str = path.split("weights_")[-1]

    if "static" in path:
        weight_str = weight_str.split("_static")[0]
    else: 
        weight_str = weight_str.split("_greedy")[0]

    if "doge" in path:
        weight_str = "doge"

    if "greedy" in path:
        weight_str = "skillit"

    if "aioli" in path:
        weight_str = "aioli"

    if "doremi" in path:
        weight_str = "doremi"

    break_steps = path.split("break_")[-1].split("_")[0]

    return weight_str, break_steps 

def save_loss(loss_dict, result_path, seed, counter, split):
    loss_file = f"{split}_seed_{seed}_checkpoint-{counter}.pkl"
    loss_path = os.path.join(result_path, loss_file)
    if isinstance(loss_dict, pd.DataFrame):
        loss_dict.to_pickle(loss_path)
    else:
        with open(loss_path, "wb") as f:
            pickle.dump(loss_dict, f)
    return loss_path


def save_weights(weights, result_path, seed, counter, all_domains, split, do_not_log=False):
    if weights is not None:
        weights /= sum(weights)
        weights_dict = {skill_idx: weights[skill_idx] for skill_idx in range(len(weights))}
        weights_file = f"{split}_seed_{seed}_proportions_checkpoint-{counter}.pkl"
        weights_path = os.path.join(result_path, weights_file)
        with open(weights_path, "wb") as f:
            pickle.dump(weights_dict, f)

        if not do_not_log:
            for i, domain in enumerate(all_domains):
                wandb.log({f"{domain}/weights": weights[i]}, step=counter)

                                
def aggregate_task_category(x):
    loss_array = np.array(x['loss'].values[0])
    metric_name = "task_loss"
    metric = loss_array.mean() # total_loss/count_loss
    names = {metric_name: metric}    
    return pd.Series(names, index=[metric_name])


def log_val_loss_per_skill(logger, loss_dict, counter, split, do_not_log=False):
    """ Logs the average loss per skill"""
    df= pd.DataFrame([{"task_idx": k, "loss": [values.item() for values in v]} for k, v in loss_dict.items()])
    df = df.groupby("task_idx").apply(lambda x: aggregate_task_category(x)).reset_index()
    logger.info(f"{split} loss: {df}")

    if not do_not_log:
        df.apply(lambda x: wandb.log({f"{x['task_idx']}/{split}_loss": x['task_loss']}, step=counter), axis=1)

    return df


class StringDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator for samples with string data in addition to tensors."""
    def __init__(self, tokenizer, string_columns, mlm):
        super().__init__(tokenizer, mlm)
        self.string_columns = string_columns
                
    def __call__(self, examples):
        tensor_examples = [{k: v for k,v in ex.items() if k not in self.string_columns} for ex in examples]
        string_examples = [{k: v for k,v in ex.items() if k in self.string_columns} for ex in examples]
        batch = super().__call__(tensor_examples)
        counts = [len(s) for s in string_examples]
        if sum(counts) != 0:
            for col in self.string_columns:
                if col in string_examples[0]: # check that the string_column exists
                    batch[col] = [ex[col] for ex in string_examples]
        return batch
    


def check_val_losses(losses):
    success = True 
    if torch.isnan(losses).any():
        nans = np.array(torch.isnan(losses))
        nan_idxs = np.where(nans == 1)[0]
        print("NaN detected in averaging loss (due to all losses in a sample being equal to 0). Setting loss equal to 0.")
        for idx in nan_idxs:
            losses[idx] = 0
        success = False 
    return losses, success 
