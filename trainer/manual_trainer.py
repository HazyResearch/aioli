import numpy as np

from tqdm import tqdm
import torch

import wandb
import sys

from torch.nn.utils import clip_grad_norm_
from .utils import (
    get_tokenized_train_dataset, 
    get_steps, 
    create_optimizer_scheduler, 
    get_train_dataloader, 
    get_tokenized_val_dataset,
    save_model_and_optimizer,
)
from .trainer import AbstractTrainer

class ManualTrainer(AbstractTrainer):
    def train(self):
        """ Training code that supports manual online data selection according to args.proportions_schedule."""
        tokenized_val = get_tokenized_val_dataset(self.args, self.validation_data)
        tokenized_test = get_tokenized_val_dataset(self.args, self.test_data)

        ckpt_steps, total_steps = get_steps(self.args)
        optimizer, lr_scheduler = create_optimizer_scheduler(self.model, self.args.lr, total_steps, self.args.lr_scheduler, self.args.end_lr, self.args.warmup_steps)

        proportions_schedule = np.array(self.args.proportions_schedule).reshape((int(len(self.args.proportions_schedule) / self.train_data.k), self.train_data.k))

        assert(len(proportions_schedule) == int(self.args.max_steps * self.args.n_epochs / self.args.update_steps))
        # get first set of skills weights from args.proportions_schedule
        weights = proportions_schedule[0]
        self.train_data.set_proportions(self.args, weights)

        tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.update_steps * self.args.batch_size)
        train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)

        progress_bar = tqdm(range(total_steps))


        self.model.zero_grad()
        logging_steps = 50
        counter = 0
        max_grad_norm = 1.0
        segment_counter = 0
        self.logger.info(f"t: {counter}, proportions: {weights/sum(weights)}. ")
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
                    self.logger.info(f"train_loss: {loss_all}")

                if counter % ckpt_steps == 0:   
                    _ = self.evaluator.evaluate(
                        tokenized_val, counter, weights,
                        split="val"
                    )              

                    
                dataloader_step += 1     
                counter += 1
                progress_bar.update(1)

                if dataloader_step == self.args.update_steps: # this is to finish this round of training 
                    segment_counter += 1
                    break

                if self.args.break_steps is not None and counter == self.args.break_steps:
                    # save model training 
                    model_path = f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt"
                    save_model_and_optimizer(self.model, optimizer, lr_scheduler, model_path)
                    sys.exit(0) 

            if counter == total_steps: # only checks this after the train_dataloader is exhausted (this is to stop training)
                break 
            
            # sample more training data according to next list of skills in args.proportions_schedule
            weights = proportions_schedule[segment_counter]
            self.logger.info(f"t: {counter}, proportions: {weights/sum(weights)}. ")
            
            self.train_data.set_proportions(self.args, weights)
            tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.update_steps*self.args.batch_size)
            train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)
            
        self.evaluator.evaluate(tokenized_val, counter, weights, 
            split="val")    

        self.evaluator.evaluate(tokenized_test, counter, weights, 
            split="test")    

        return self.model 