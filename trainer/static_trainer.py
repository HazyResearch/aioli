from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import wandb 
from torch.utils.data import IterableDataset
import sys
from pathlib import Path

from .utils import (
    get_tokenized_train_dataset, 
    get_tokenized_val_dataset,
    get_steps, 
    create_optimizer_scheduler, 
    get_train_dataloader,
    save_model_and_optimizer,
    load_model_and_optimizer,
    get_n_data)
from .trainer import AbstractTrainer 

import sys


class StaticTrainer(AbstractTrainer):
    def train(self):
        """Standard Pytorch training and evaluation code without any online sampling."""

        tokenized_val = get_tokenized_val_dataset(self.args, self.validation_data)
        tokenized_test = get_tokenized_val_dataset(self.args, self.test_data)

        n_data = get_n_data(self.args)
        tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, n_data)
        train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)   

        ckpt_steps, total_steps = get_steps(self.args)

        optimizer, lr_scheduler = create_optimizer_scheduler(self.model, self.args.lr, total_steps, self.args.lr_scheduler, self.args.end_lr, self.args.warmup_steps)

        break_steps_from_resume_model = None
        if self.args.resume_model_path is not None:
            self.model, optimizer, lr_scheduler, total_steps = load_model_and_optimizer(
                self.model, optimizer, lr_scheduler, self.args.remaining_steps, self.args.resume_model_path)
            break_steps_from_resume_model = int(self.args.resume_model_path.split("break_")[-1].split("_")[0]) 
            total_steps += break_steps_from_resume_model    

        progress_bar = tqdm(range(total_steps))
        logging_steps = 50
        counter = 0
        max_grad_norm = 1.0
        self.model.zero_grad()

        num_epochs = 1 if isinstance(tokenized_train, IterableDataset) else self.args.n_epochs
            
        for _ in range(num_epochs):
            for i, batch in enumerate(train_dataloader):

                if break_steps_from_resume_model is not None and counter < break_steps_from_resume_model:
                    # roll forward the data 
                    counter += 1 
                    progress_bar.update(1)
                    continue 

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


                if (break_steps_from_resume_model is None and counter % ckpt_steps == 0 and counter != self.args.break_steps) or (
                    break_steps_from_resume_model is not None and counter % ckpt_steps == 0 and counter != break_steps_from_resume_model):
                    self.evaluator.evaluate(
                        tokenized_val, counter, None,
                        split="val")   


                counter += 1
                progress_bar.update(1)

                if self.args.break_steps is not None and counter == self.args.break_steps:
                    # save model training 
                    df = self.evaluator.evaluate(
                        tokenized_val, counter, None,
                        split="val")   

                    if not self.args.do_not_save:
                        model_path = Path(f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt")
                        save_model_and_optimizer(self.model, optimizer, lr_scheduler, model_path)
                        sys.exit(0) 
                    else:
                        return loss_all, df


                if counter == total_steps:
                    # exit after total_steps foward passes/gradient updates
                    # this is needed for iterable datasets. For mapped datasets, will only be run at the end of training anyways
                    break

    
        self.evaluator.evaluate(tokenized_val, counter, None, 
            split="val")
        
        self.evaluator.evaluate(tokenized_test, counter, None,
            split="test")

        return self.model