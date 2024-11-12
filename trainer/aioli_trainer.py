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

import os 

def save_aioli_matrices(matrices, seed, result_path):
    print(f"Aioli matrices are: {matrices}")
    drm_file = os.path.join(result_path, f"seed_{seed}_aioli_matrices.npy")
    np.save(drm_file, matrices)


class AioliTrainer(AbstractTrainer):

    def skillit_subroutine(self, weights, graph, tokenized_val):
        """
            Replicates exact skill-it training procedure
        """
        self.train_data.set_proportions(self.args, weights)
        tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.skillit_update_steps * self.args.batch_size)
        train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)
        
        weights_init = np.ones(self.train_data.k)


        all_losses = []
        self.model.zero_grad()
        counter = 0
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
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()

                if counter % self.logging_steps == 0:
                    wandb.log({"train_loss": loss_all}, step=counter)
                    
                if counter % self.ckpt_steps == 0:                    
                    df = self.evaluator.evaluate(
                        tokenized_val, counter, weights, None, split="val"
                    )  
                         
                    all_losses.append(df.task_loss.values)
                    
                dataloader_step += 1     
                counter += 1
                self.progress_bar.update(1)

                if counter == self.args.prior_steps:
                    return counter, weights


                if self.args.break_steps is not None and counter == self.args.break_steps:
                    # save model training 
                    self.evaluator.evaluate(
                        tokenized_val, counter, weights, None,
                        split="val"
                    )

                    model_path = f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt"
                    save_model_and_optimizer(self.model, self.optimizer, self.lr_scheduler, model_path)
                    sys.exit(0) 
                        
                if dataloader_step == self.args.skillit_update_steps:
                    break
            
            
            # update skills mixture 
            idx = len(all_losses)            
            eta_t = self.args.skillit_eta
            if self.args.skillit_window >= 0:
                loss_arr = np.array(all_losses[max(0, idx - self.args.skillit_window): idx]).sum(axis=0)

                self.logger.info(f"loss are: {loss_arr}")
        
                weights = np.multiply(weights_init, np.exp(eta_t * graph.dot(loss_arr)))
                self.logger.info(f"weights: {weights}")


            self.logger.info(f"t={counter}, new data distribution={weights/sum(weights)}")
            
            # create new dataset for next round
            self.train_data.set_proportions(self.args, weights)
            tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.update_steps*self.args.batch_size)
            train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)


    def learn_params_subroutine(self, loss_0, optimizer, lr_scheduler, tokenized_val):
        previous_loss = loss_0.copy()
        new_graph = np.zeros((self.args.k, self.args.k))

        mixture_order = np.concatenate( [np.random.permutation(self.args.k) for i in range(self.args.lp_rounds)] )
        for onehot_idx in mixture_order:
            weights = np.ones(self.args.k) * (1 - self.args.one_hot_factor)/(self.args.k-1)
            weights[onehot_idx] = self.args.one_hot_factor
            self.train_data.set_proportions(self.args, weights)

            mapped_train = True if self.args.one_hot_factor != 1 else False
            tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.lp_steps * self.args.batch_size, mapped_train=mapped_train)
            train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)

            counter = 0
            for _, batch in enumerate(train_dataloader):
                self.model.train()

                input_ids = batch['input_ids'].cuda()
                labels = batch['labels'].cuda()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss_all = torch.mean(loss)
                loss_all.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                self.model.zero_grad()
                
                counter += 1
                if counter == self.args.lp_steps:
                    break

            # validate, compute loss difference
            df = self.evaluator.evaluate(
                tokenized_val, counter, weights, split="val", do_not_log=True
            )
            # make sure losses are in the right order  
            new_loss = np.array([df[df.task_idx == skill].task_loss.values[0] for skill in self.train_data.skills])

            new_graph[:, onehot_idx] += previous_loss - new_loss
            previous_loss = new_loss 


        new_graph /= self.args.lp_rounds 

        if self.args.one_hot_factor != 1 and not self.args.aioli_diagonal:
            print(f"New graph before: {new_graph}")
            weight_matrix = np.zeros((self.args.k, self.args.k))
            for i in range(self.args.k):
                weight_row = np.ones(self.args.k) * (1 - self.args.one_hot_factor)/(self.args.k-1)
                weight_row[i] = self.args.one_hot_factor
                weight_matrix[i] = weight_row

            A = np.zeros((self.args.k, self.args.k))
            for i, row in enumerate(new_graph):
                A[i] = np.linalg.solve(weight_matrix, row)
            
            new_graph = A
            print(f"New graph after: {new_graph}")

        elif self.args.one_hot_factor != 1 and self.args.aioli_diagonal:
            print(f"New graph before: {new_graph}")

            A = np.zeros((self.args.k, self.args.k))
            for i in range(self.args.k):
                A[i, i] = new_graph[i, i] / self.args.one_hot_factor
            
            new_graph = A
            print(f"New graph after: {new_graph}")            
        
        return new_graph
                    
    
    def train(self):
        """ Aioli algorithm.
        
            Key hyperparameters:
                - aioli_prior: if set, we start by running aioli_prior weights for prior_steps.
                - prior_steps: how many steps to run the prior (to simulate this transfer setting)
                - eta: softmax temperature hyperparameter 
                - lp_rounds: number of sweeps through the k dataset 
                - lp_steps: number of contiguous batches to take for each dataset
                - update_steps: how many steps to update weights
                - aioli_normalize_A: whether or not to normalize the graph matrix before softmaxxing 
        """
        if self.args.k is None and len(self.args.slice_list) > 0:
            self.args.k = len(self.args.slice_list)

        tokenized_val = get_tokenized_val_dataset(self.args, self.validation_data)
        tokenized_test = get_tokenized_val_dataset(self.args, self.test_data)

        self.ckpt_steps, self.total_steps = get_steps(self.args)
        self.optimizer, self.lr_scheduler = create_optimizer_scheduler(self.model, self.args.lr, self.total_steps, self.args.lr_scheduler, self.args.end_lr, self.args.warmup_steps)

        if self.args.graph_path is not None:
            graph = np.load(self.args.graph_path).astype(float)

        if self.args.aioli_prior is not None:
            assert self.args.prior_steps is not None
            weights_init = np.array(self.args.aioli_prior)
            self.logger.info(f"Setting initial weights to be {self.args.aioli_prior}, training for {self.args.prior_steps} steps.")
        else:
            weights_init = np.ones(self.train_data.k)

        if self.args.graph_path is not None:
            weights = np.multiply(weights_init, np.exp(self.args.skillit_eta * graph.dot(weights_init)))
            self.logger.info(f"Setting initial weights to be {weights}, training for {self.args.prior_steps} steps.")
        else:
            weights = weights_init

        weights_init = np.ones(self.train_data.k) 
    
        lp_duration = self.args.lp_rounds * self.args.lp_steps * self.args.k # the \delta fraction of a round in LearnParams
        n_data = self.args.update_steps * self.args.batch_size - lp_duration

        # universal settings for training
        self.progress_bar = tqdm(range(self.total_steps))
        self.logging_steps = 50
        self.max_grad_norm = 1.0

        if self.args.aioli_prior is not None:
            dynamic_steps = self.total_steps - self.args.prior_steps
            self.logger.info(f"Remaining dynamic steps: {dynamic_steps}")
            assert dynamic_steps % self.args.update_steps == 0, f"Remaining steps ({dynamic_steps}) should be divisible by update steps ({self.args.update_steps})"
            n_data -= (self.args.batch_size * self.args.prior_steps)
        else:
            dynamic_steps = self.total_steps


        self.model.zero_grad()
        counter = 0

        if self.args.aioli_prior is not None:

            self.train_data.set_proportions(self.args, weights)
            tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, self.args.batch_size * self.args.prior_steps)
            train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)

            self.logger.info("Running static mixture!")
            for _, batch in enumerate(train_dataloader):
                self.model.train()

                input_ids = batch['input_ids'].cuda()
                labels = batch['labels'].cuda()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss_all = torch.mean(loss)
                loss_all.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()
                    
                if counter % self.logging_steps == 0:
                    wandb.log({"train_loss": loss_all}, step=counter)
                    
                counter += 1
                self.progress_bar.update(1)


                if counter % self.ckpt_steps == 0:
                    self.evaluator.evaluate(
                        tokenized_val, counter, weights, 
                        split="val"
                    )

                if self.args.break_steps is not None and counter == self.args.break_steps:
                    self.logger.warning("We are breaking in the static part of the training!")
                    # save model training 
                    self.evaluator.evaluate(
                        tokenized_val, counter, weights, 
                        split="val"
                    )

                    model_path = f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt"
                    save_model_and_optimizer(self.model, self.optimizer, self.lr_scheduler, model_path)
                    #sys.exit(0) WE DO NOT EXIT for efficiency's sake 

                if counter == self.args.prior_steps:
                    break 
        elif self.args.graph_path is not None:
            counter, weights = self.skillit_subroutine(weights, graph, tokenized_val)

        print(f"Counter is {counter}, weights are ={weights}")
        num_iterations = -(-dynamic_steps // self.args.update_steps)  # Round up division, ckpt_steps is the same as our update steps
        self.logger.info(f"Running dynamic stage for {num_iterations} iterations!")
        self.all_graphs = []
        for i in range(num_iterations):
            df = self.evaluator.evaluate(
                tokenized_val, counter, weights, split="val"
            )  
            

            loss_0 = np.array([df[df.task_idx == skill].task_loss.values[0] for skill in self.train_data.skills])
            
            graph = self.learn_params_subroutine(loss_0, self.optimizer, self.lr_scheduler, tokenized_val)
            self.all_graphs.append(graph)
            
            self.logger.info(f"LearnParams done. New graph is {graph}")
            counter += lp_duration 
            self.progress_bar.update(lp_duration)
            
            if self.args.aioli_normalize_A:
                min_entry = graph.min()
                if min_entry < 0:
                    graph -= min_entry 
                    self.logger.info(f"Rescaled graph is {graph}, previous min entry was {min_entry}")

                graph /= graph.sum()
                self.logger.info(f"Graph after normalization: {graph}")

            # update skills mixture 
            if self.args.ema is not None:
                if i == 0:
                    ema_graph = graph
                else:
                    ema_graph = (1-self.args.ema) * graph + self.args.ema * ema_graph
                self.logger.info(f"Applying ema, smoothed graph is {ema_graph}")
                weights = np.multiply(weights_init, np.exp(self.args.eta * ema_graph.sum(axis=0)))
            else:
                if i == 0:
                    weights = np.multiply(weights_init, np.exp(self.args.eta * graph.sum(axis=0)))
                else:
                    weights = np.multiply(weights, np.exp(self.args.eta * graph.sum(axis=0)))

            self.logger.info(f"t={counter}, new data distribution={weights/sum(weights)}. ")


            self.train_data.set_proportions(self.args, weights)
            tokenized_train = get_tokenized_train_dataset(self.args, self.train_data, n_data)
            train_dataloader = get_train_dataloader(self.tokenizer, tokenized_train, self.args.batch_size)

            for _, batch in enumerate(train_dataloader):
                self.model.train()

                input_ids = batch['input_ids'].cuda()
                labels = batch['labels'].cuda()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss_all = torch.mean(loss)
                loss_all.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()
                
                if counter % self.logging_steps == 0:
                    wandb.log({"train_loss": loss_all}, step=counter)
                    
                counter += 1
                self.progress_bar.update(1)


                if counter % self.ckpt_steps == 0:
                    self.evaluator.evaluate(
                        tokenized_val, counter, weights,
                        split="val"
                    )


                if self.args.break_steps is not None and counter == self.args.break_steps:
                    # save model training 
                    save_aioli_matrices(self.all_graphs, self.args.selection_seed, self.evaluator.result_path)


                    self.evaluator.evaluate(
                        tokenized_val, counter, weights, 
                        split="val"
                    )

                    model_path = f"./saved_model_optim/{self.run_name}_seed_{self.args.selection_seed}.pt"
                    save_model_and_optimizer(self.model, self.optimizer, self.lr_scheduler, model_path)
                    sys.exit(0) 
                        
                if counter % self.args.update_steps == 0:
                    break
            
            if counter == self.total_steps:
                break 
 
        self.evaluator.evaluate(
            tokenized_val, counter, weights, split="val"
        )      

        self.evaluator.evaluate(
            tokenized_test, counter, weights, split="test"
        )

            
        return self.model