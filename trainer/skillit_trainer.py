import numpy as np
from tqdm import tqdm
import torch
import wandb
import torch
from torch.nn.utils import clip_grad_norm_

import sys 


from .utils import (
    get_steps, 
    get_tokenized_val_dataset,
    create_optimizer_scheduler, 
    get_train_dataloader, 
    get_tokenized_train_dataset, 
    save_model_and_optimizer,
)

from .trainer import AbstractTrainer

class SkillitTrainer(AbstractTrainer):
    def train(self):
        """ Skill-It online data selection, plus algorithmic variations."""    
        tokenized_val = get_tokenized_val_dataset(self.args, self.validation_data)
        tokenized_test = get_tokenized_val_dataset(self.args, self.test_data)

        ckpt_steps, total_steps = get_steps(self.args)
        optimizer, lr_scheduler = create_optimizer_scheduler(self.model, self.args.lr, total_steps, self.args.lr_scheduler, self.args.end_lr, self.args.warmup_steps)
            
        all_losses = []
        
        if self.args.graph_path is not None:
            graph = np.load(self.args.graph_path).astype(float)
            n, m = graph.shape
            for i in range(n):
                for j in range(m):
                    if i != j and graph[i, j] == 1:
                        graph[i, j] = 0.5       
        else:
            graph = np.eye(self.args.k)
        
        self.logger.info(f"Using dependency graph:\n{graph}")

            
        weights_init = np.ones(graph.shape[0])
        self.logger.info(f"weights init are {weights_init}")
    
    
        weights = np.multiply(weights_init, np.exp(self.args.eta * graph.dot(weights_init)))

        self.train_data.set_proportions(self.args, weights)
        tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.update_steps * self.args.batch_size)
        train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)
        
        self.model.zero_grad()
        logging_steps = 50
        counter = 0
        max_grad_norm = 1.0
        progress_bar = tqdm(range(total_steps))
        self.logger.info(f"t={counter}, new data distribution={weights/sum(weights)}. ")
        while True:    
            dataloader_step = 0
            for idx, batch in enumerate(train_dataloader):
                self.model.train()

                input_ids = batch['input_ids'].cuda()
                labels = batch['labels'].cuda()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss_all = torch.mean(loss)
                loss_all.backward()
                clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                self.model.zero_grad()
                if counter % logging_steps == 0:
                    wandb.log({"train_loss": loss_all}, step=counter)
                    
                if counter % ckpt_steps == 0:                    
                    df = self.evaluator.evaluate(
                        tokenized_val, counter, weights, split="val"
                    )  
                    
                    if self.args.task_name == "ni":
                        tokenized_val, _ = self.validation_data.get_tokenized_dataset()          
                                             
                    all_losses.append(df.task_loss.values)
                    
                dataloader_step += 1     
                counter += 1
                progress_bar.update(1)

                if self.args.break_steps is not None and counter == self.args.break_steps:
                    # save model training 
                    self.evaluator.evaluate(
                        tokenized_val, counter, weights,
                        split="val"
                    )

                    model_path = f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt"
                    save_model_and_optimizer(self.model, optimizer, lr_scheduler, model_path)
                    sys.exit(0) 
                        
                if dataloader_step == self.args.update_steps:
                    break
            
            if counter == total_steps:
                break 
            
            # update skills mixture 
            idx = len(all_losses)            
            eta_t = self.args.eta
            if self.args.skillit_window >= 0:
                loss_arr = np.array(all_losses[max(0, idx - self.args.skillit_window): idx]).sum(axis=0)

                self.logger.info(f"loss are: {loss_arr}")
            
                weights = np.multiply(weights_init, np.exp(eta_t * graph.dot(loss_arr)))
                self.logger.info(f"weights: {weights}")

            else:
                raise NotImplementedError("skillit_window must be > 0, standard multiplicative weights not supported.")

            self.logger.info(f"t={counter}, new data distribution={weights/sum(weights)}")
            
            # create new dataset for next round
            self.train_data.set_proportions(self.args, weights)
            tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.update_steps*self.args.batch_size)
            train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)
                        
        self.evaluator.evaluate(
            tokenized_val, counter, weights, split="val"
        )      

        self.evaluator.evaluate(
            tokenized_test, counter, weights, split="test"
        )

        return self.model