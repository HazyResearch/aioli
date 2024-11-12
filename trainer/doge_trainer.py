"""
Code adapted from https://github.com/Olivia-fsm/DoGE.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, RandomSampler

import pickle
import wandb
from tqdm import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict

from pathlib import Path
import sys

import os 

from .trainer import AbstractTrainer
from .utils import (
    get_tokenized_train_dataset, 
    get_steps, 
    create_optimizer_scheduler, 
    get_train_dataloader,
    get_tokenized_val_dataset,
    get_n_data,
    save_model_and_optimizer
)

from evaluator.utils import StringDataCollator

        
def get_flat_grad(model, tgt_params_ls=None):
    ''' Get flattened gradient vectors for all layers. '''
    full_grad_concat = None
    for p_name, p in model.named_parameters():
        if tgt_params_ls is not None and p_name not in tgt_params_ls:
            continue
        flat_grad = p.grad.detach().flatten()
        if full_grad_concat is not None:
            full_grad_concat = torch.concat([full_grad_concat, flat_grad])
        else:
            full_grad_concat = flat_grad
    return full_grad_concat

def get_grad_dict(model):
    full_grad_dict = {}
    for p_name, p in model.named_parameters():
        full_grad_dict[p_name] = p.grad
    return full_grad_dict

def add_model_grad_ls(model, domain_full_grad_dict, dw=None):
    # set gradients back into the .grad variables
    if dw is None or type(domain_full_grad_dict)==dict:
        add_model_grad(model, domain_full_grad_dict)
    for p_name, p in model.named_parameters():
        for idx,v in enumerate(dw):
            if domain_full_grad_dict[idx] is not None: # skips domains that aren't in the batch
                if p.grad is None:
                    p.grad = domain_full_grad_dict[idx][p_name]*v
                else:
                    p.grad += domain_full_grad_dict[idx][p_name]*v

def add_model_grad(model, domain_full_grad_dict):
    for p_name, p in model.named_parameters():
        if p.grad is None:
            p.grad = domain_full_grad_dict[p_name]
        else:
            p.grad += domain_full_grad_dict[p_name]


def log_weights(train_dw, avg_dw, all_domains, counter):
    for i, domain in enumerate(all_domains):
        wandb.log({f"{domain}/train_dw": train_dw[i]}, step=counter)
        wandb.log({f"{domain}/avg_dw": avg_dw[i]/(counter+1)}, step=counter)


def save_avg_weights(avg_dw, steps, seed, result_path):
    avg_weights_dict = {
        i: avg_dw[i]/steps for i in range(len(avg_dw))
    }

    print(f"Average final domain weights are: {avg_weights_dict}")

    avg_weights_file = f"seed_{seed}_doge_avg_dw.pkl"
    avg_weights_path = os.path.join(result_path, avg_weights_file)
    with open(avg_weights_path, "wb") as f:
            pickle.dump(avg_weights_dict, f)

def save_doge_matrices(matrices, seed, result_path, break_step_avg):
    print(f"Doge matrices are: {matrices}")
    doge_file = os.path.join(result_path, f"seed_{seed}_avg_{break_step_avg}_doge_matrices.npy")
    np.save(doge_file, matrices)

def make_val_dataloader(tokenized_data, tokenizer, batch_size):
    string_columns = ["skill"]
    data_collator = StringDataCollator(tokenizer, string_columns, mlm=False)

    sampler = RandomSampler(tokenized_data, replacement=True, num_samples=int(1e100))

    dataloader = DataLoader(
        tokenized_data,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )
    return dataloader


class DogeTrainer(AbstractTrainer):

    def update_domain_weights(self, pertoken_losses, val_loss, token_masks, train_domain_ids, val_domain_ids, counter, compute_full_matrix, reweight_eps=0.0, dw_min=0.0, dw_max=5.0):
        if isinstance(train_domain_ids, list):
            train_domain_ids = np.array(train_domain_ids)
        if isinstance(val_domain_ids, list):
            val_domain_ids = np.array(val_domain_ids)

        full_grad_dicts = []
        domain_losses = {}

        flat_grad_mat = torch.zeros((self.train_data.k + 1, self.model.num_parameters()), dtype=torch.float32)

        for i, domain_id in enumerate(self.all_domains):
            # compute losses per domain and add to buffer
            domain_mask = (train_domain_ids == domain_id)

            if domain_mask.sum() == 0:
                domain_losses[domain_id] = None
            else:
                curr_domain_losses = pertoken_losses[(token_masks[:, :self.args.context_length-1] * domain_mask.reshape(-1, 1)).bool()].mean()
                domain_losses[domain_id] = curr_domain_losses

            
        if compute_full_matrix:
            val_domain_losses = {}
            for i, domain_id in enumerate(self.all_domains):
                domain_mask = (val_domain_ids == domain_id)
                if domain_mask.sum() == 0:
                    val_domain_losses[domain_id] = None
                else:
                    curr_domain_losses = val_loss[domain_mask].mean()
                    val_domain_losses[domain_id] = curr_domain_losses
        else:
            domain_losses['mix'] = val_loss.mean()

        for i, (domain_id, loss) in enumerate(domain_losses.items()):
            if loss is None:
                full_grad_dicts.append(None)
            else:
                loss.backward(retain_graph=True)
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                domain_flat_grad = get_flat_grad(self.model)
                domain_full_grad_dict = get_grad_dict(self.model)

                flat_grad_mat[i] = domain_flat_grad.float()
                full_grad_dicts.append(domain_full_grad_dict)
            
                self.model.zero_grad()


        if compute_full_matrix:
            val_flat_grad_mat = torch.zeros((self.train_data.k, self.model.num_parameters()), dtype=torch.float32)
            for i, (domain_id, loss) in enumerate(val_domain_losses.items()):
                if loss is None:
                    continue
                else:
                    loss.backward(retain_graph=True)
                    clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    domain_flat_grad = get_flat_grad(self.model)
                    val_flat_grad_mat[i] = domain_flat_grad.float()
                
                    self.model.zero_grad()


            doge_matrix = flat_grad_mat[:-1] @ val_flat_grad_mat.T
            self.doge_matrices.append(doge_matrix)
            scores_mat = doge_matrix.mean(dim=0).reshape(-1, 1)
        else:
            scores_mat = flat_grad_mat[:-1] @ flat_grad_mat[-1].reshape(1, -1).T

        lr = self.lr_scheduler.get_last_lr()[0]
        scores = lr * (scores_mat.sum(dim=-1))
        avg_norm = flat_grad_mat[:-1].norm(dim=-1).mean()
        scores = scores / (avg_norm + 1e-6)
        scores = torch.clip(scores, min=lr*dw_min, max=lr*dw_max)

        dw_prev = self.train_dw
        log_dw_new = torch.log(dw_prev) + scores / self.args.doge_mu
        dw_new = F.softmax(log_dw_new, dim=-1)
        dw_new = (1-reweight_eps) * dw_new + reweight_eps / len(dw_new)
        self.train_dw = dw_new

        self.avg_dw += dw_new

        log_weights(self.train_dw, self.avg_dw, self.all_domains, counter)

        add_model_grad_ls(self.model, full_grad_dicts[:-1] if not compute_full_matrix else full_grad_dicts, dw=self.train_dw)


    def train_step(self, inputs, val_inputs, counter, compute_full_matrix=False):
        self.model.train()

        input_ids = inputs['input_ids'].cuda()
        labels = inputs['labels'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        loss = self.model(input_ids = input_ids, attention_mask=attention_mask, labels = labels).loss
        loss = loss.reshape(self.args.batch_size, -1)

        input_ids = val_inputs['input_ids'].cuda()
        labels = val_inputs['labels'].cuda()
        attention_mask = val_inputs['attention_mask'].cuda()



        val_loss = self.model(input_ids = input_ids, attention_mask=attention_mask, labels = labels).loss
        val_loss = val_loss.reshape(self.args.doge_val_batch_size, -1)


        train_domain_ids = inputs['skill']
        val_domain_ids = val_inputs['skill']

        self.update_domain_weights(loss, val_loss, inputs['attention_mask'], train_domain_ids, val_domain_ids, counter, compute_full_matrix)

        return loss.detach() 
    
    def train(self):
        tokenized_val = get_tokenized_val_dataset(self.args, self.validation_data)
        tokenized_test = get_tokenized_val_dataset(self.args, self.test_data)

        val_dataloader = make_val_dataloader(tokenized_val, self.tokenizer, self.args.doge_val_batch_size)

        n_data = get_n_data(self.args)
        tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, n_data)
        train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)   

        ckpt_steps, total_steps = get_steps(self.args)
        self.optimizer, self.lr_scheduler = create_optimizer_scheduler(self.model, self.args.lr, total_steps, self.args.lr_scheduler, self.args.end_lr, self.args.warmup_steps)

        self.train_dw = torch.ones(self.train_data.k, dtype=torch.float32) / self.train_data.k # initial domain weights are uniform
        self.avg_dw = torch.zeros(self.train_data.k, dtype=torch.float32)

        self.all_domains = self.train_data.skills

        if len(self.all_domains.shape) > 1:
            self.all_domains = self.all_domains[0]
        
        progress_bar = tqdm(range(total_steps))
        counter = 0
        logging_steps = 50
        self.max_grad_norm = 1.0
        self.model.zero_grad()
        
        num_epochs = 1 if isinstance(tokenized_train, IterableDataset) else self.args.n_epochs

        self.doge_matrices = []
        
        for _ in range(num_epochs):
            for i, batch in enumerate(train_dataloader):
                val_inputs = next(iter(val_dataloader))

                if self.args.break_steps is not None and counter > self.args.break_steps - self.args.doge_break_step_average:
                    loss = self.train_step(batch, val_inputs, counter, compute_full_matrix=True)
                else:
                    loss = self.train_step(batch, val_inputs, counter)

                self.optimizer.step()
                self.lr_scheduler.step()


                self.model.zero_grad()

                if counter % logging_steps == 0:
                    wandb.log({"train_loss": loss.mean()}, step=counter)
                    self.logger.info(f"train_loss: {loss.mean()}")

                if counter % ckpt_steps == 0:                
                    self.evaluator.evaluate(tokenized_val, counter, self.train_dw,
                                            )
                    #
                    self.logger.info(f"Train_DW: {self.train_dw}")

                counter += 1
                progress_bar.update(1)

                if self.args.break_steps is not None and counter == self.args.break_steps:
                    # save model training 
                    # save doge matrices
                    self.doge_matrices = torch.stack(self.doge_matrices).numpy()
                    save_doge_matrices(self.doge_matrices, self.args.selection_seed, self.evaluator.result_path, self.args.doge_break_step_average)

                    self.evaluator.evaluate(
                        tokenized_val, counter, None,
                        split="val")   

                    model_path = Path(f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt")
                    save_model_and_optimizer(self.model, self.optimizer, self.lr_scheduler, model_path)
                    sys.exit(0) 


                if counter == total_steps:
                    break
    
        self.evaluator.evaluate(tokenized_val, counter, self.train_dw, split="val")
        self.evaluator.evaluate(tokenized_test, counter, self.train_dw, split="test")

        log_weights(self.train_dw, self.avg_dw, self.all_domains, counter)
        save_avg_weights(self.avg_dw, total_steps, self.args.selection_seed, self.evaluator.result_path)

        return self.model