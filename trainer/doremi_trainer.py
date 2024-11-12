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

        

def log_weights(train_dw, avg_dw, all_domains, counter):
    for i, domain in enumerate(all_domains):
        wandb.log({f"{domain}/train_dw": train_dw[i]}, step=counter)
        wandb.log({f"{domain}/avg_dw": avg_dw[i]}, step=counter)


def save_avg_weights(avg_dw, seed, result_path):
    avg_weights_dict = {
        i: avg_dw[i] for i in range(len(avg_dw))
    }

    print(f"Average final domain weights are: {avg_weights_dict}")

    avg_weights_file = f"seed_{seed}_doremi_avg_dw.pkl"
    avg_weights_path = os.path.join(result_path, avg_weights_file)
    with open(avg_weights_path, "wb") as f:
            pickle.dump(avg_weights_dict, f)

def save_drm_matrices(matrices, seed, result_path, break_step_avg):
    print(f"Doremi matrices are: {matrices.mean(axis=0)}")
    drm_file = os.path.join(result_path, f"seed_{seed}_avg_{break_step_avg}_drm_matrices.npy")
    np.save(drm_file, matrices)


class DoremiTrainer(AbstractTrainer):

    def update_domain_weights(self, domain_losses, ref_domain_losses, counter, compute_full_matrix=False, reweight_eps=1e-3):
        '''
        NTS: backprop also performed here.
        TODO: implement case where train_ids != test_ids
        '''
        self.dw_update_steps += 1

        if compute_full_matrix:
            excess_losses = np.array([domain_losses[i].clone().detach().float().cpu().numpy() - ref_domain_losses[i].clone().detach().float().cpu().numpy() for i in range(len(self.all_domains))])
            self.drm_matrices.append(np.diag(excess_losses))


        excess_losses = torch.tensor([domain_losses[i] - ref_domain_losses[i] for i in range(len(self.all_domains))])
        for i in range(len(excess_losses)):
            if excess_losses[i] == 0:
                excess_losses[i] = self.perdomain_scores[i]


        excess_losses = torch.clip(excess_losses, min=0.0) # bounding it by 0
        self.perdomain_scores = excess_losses.detach().cpu().tolist()
        
        log_new_train_dw = torch.log(self.train_dw) + self.args.doremi_mu * excess_losses
        log_new_train_dw = log_new_train_dw - torch.logsumexp(log_new_train_dw, dim=0) # softmax normalization
        # smoothing
        dw_new = (1-reweight_eps) * torch.exp(log_new_train_dw) + reweight_eps / len(log_new_train_dw)
        
        self.train_dw = dw_new
        self.avg_dw += dw_new

        log_weights(self.train_dw, self.avg_dw/self.dw_update_steps, self.all_domains, counter)

        for i in range(len(self.all_domains)):
            curr_domain_loss = domain_losses[i] * dw_new[i] # loss weighted by dw
            if curr_domain_loss > 0:
                curr_domain_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)



    def train_step(self, inputs, counter, compute_full_matrix=False):
        self.model.train()

        loss_all = 0
        effective_domains_per_batch =0

        self.domain_losses = [torch.tensor(0.0) for _ in range(len(self.all_domains)) ]
        self.ref_domain_losses = [torch.tensor(0.0) for _ in range(len(self.all_domains)) ]

        skills = inputs['skill']
        if isinstance(skills, list):
            skills = np.array(skills)
    
        for i, domain_id in enumerate(self.all_domains):
            # compute losses per domain and add to buffer
            domain_mask = (skills == domain_id)
            if domain_mask.sum() == 0:
                continue 
            new_inputs = inputs['input_ids'][domain_mask].cuda()
            new_labels = inputs['labels'][domain_mask].cuda()
            new_attention_mask = inputs['attention_mask'][domain_mask].cuda()

            loss = self.model(input_ids=new_inputs, attention_mask=new_attention_mask, labels=new_labels).loss
            loss = loss.reshape(sum(domain_mask), -1).mean()
            ref_loss = self.reference_model(input_ids=new_inputs, attention_mask=new_attention_mask, labels=new_labels).loss
            ref_loss = ref_loss.reshape(sum(domain_mask), -1).mean()

            self.domain_losses[i] = loss 
            self.ref_domain_losses[i] = ref_loss
            loss_all += loss.detach().cpu()
            effective_domains_per_batch += 1

        self.grad_acc_step += 1

        if self.grad_acc_step == self.args.gradient_accumulation_steps:
            if self.args.gradient_accumulation_steps > 1:
                self.domain_losses = [l / self.args.gradient_accumulation_steps for l in self.domain_losses]
                self.ref_domain_losses = [l / self.args.gradient_accumulation_steps for l in self.ref_domain_losses]
            self.update_domain_weights(self.domain_losses, self.ref_domain_losses, counter, compute_full_matrix)

            self.grad_acc_step = 0 

        return loss_all / (self.args.gradient_accumulation_steps * effective_domains_per_batch)
    
    def train(self):
        from utils import load_model

        assert self.args.doremi_reference_model_path is not None, "Doremi needs a reference model."
        self.reference_model = load_model(self.args.model_name, self.args.doremi_reference_model_path).cuda()
        self.reference_model.eval() # switch reference model to eval mode

        tokenized_val = get_tokenized_val_dataset(self.args, self.validation_data)
        tokenized_test = get_tokenized_val_dataset(self.args, self.test_data)

        n_data = get_n_data(self.args)
        tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, n_data)
        train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)   

        ckpt_steps, total_steps = get_steps(self.args)
        self.optimizer, self.lr_scheduler = create_optimizer_scheduler(self.model, self.args.lr, total_steps, self.args.lr_scheduler, self.args.end_lr, self.args.warmup_steps)

        self.train_dw = torch.ones(self.train_data.k, dtype=torch.float32) / self.train_data.k # initial domain weights are uniform
        self.avg_dw = torch.zeros(self.train_data.k, dtype=torch.float32)

        self.all_domains = self.train_data.skills
        self.perdomain_scores = torch.zeros(self.train_data.k, dtype=torch.float)+1e-6

        self.dw_update_steps = 0

        if len(self.all_domains.shape) > 1:
            self.all_domains = self.all_domains[0]
        
        progress_bar = tqdm(range(total_steps))
        counter = 0
        logging_steps = 50
        self.max_grad_norm = 1.0
        self.model.zero_grad()

        self.grad_acc_step = 0

        
        num_epochs = 1 if isinstance(tokenized_train, IterableDataset) else self.args.n_epochs

        self.drm_matrices = []
        
        for _ in range(num_epochs):
            for i, batch in enumerate(train_dataloader):

                if self.args.break_steps is not None and counter > self.args.break_steps - self.args.doge_break_step_average:
                    loss = self.train_step(batch, counter, compute_full_matrix=True)
                else:
                    loss = self.train_step(batch, counter)

                self.optimizer.step()
                self.lr_scheduler.step()

                self.model.zero_grad()

                if counter % logging_steps == 0:
                    wandb.log({"train_loss": loss.mean()}, step=counter)
                    self.logger.info(f"train_loss: {loss.mean()}")

                if counter % ckpt_steps == 0:                
                    self.evaluator.evaluate(tokenized_val, counter, self.train_dw, 
                                           )
                    self.logger.info(f"Train_DW: {self.train_dw}")


                if self.args.break_steps is not None and counter == self.args.break_steps:
                    self.drm_matrices = np.array(self.drm_matrices)
                    save_drm_matrices(self.drm_matrices, self.args.selection_seed, self.evaluator.result_path, self.args.doge_break_step_average)

                    self.evaluator.evaluate(
                        tokenized_val, counter, None, 
                        split="val")   

                    model_path = Path(f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt")
                    save_model_and_optimizer(self.model, self.optimizer, self.lr_scheduler, model_path)
                    sys.exit(0) 

                counter += 1
                progress_bar.update(1)

                if counter == total_steps:
                    break
    
        self.evaluator.evaluate(tokenized_val, counter, self.train_dw, split="val")
        self.evaluator.evaluate(tokenized_test, counter, self.train_dw, split="test")

        log_weights(self.train_dw, self.avg_dw/self.dw_update_steps, self.all_domains, counter)
        save_avg_weights(self.avg_dw/self.dw_update_steps, self.args.selection_seed, self.evaluator.result_path)

        return self.model