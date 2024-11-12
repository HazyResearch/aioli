import os
import numpy as np
import torch 

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm


from .utils import (
    StringDataCollator, 
    save_loss, 
    save_weights, 
    log_val_loss_per_skill, 
    parse_resumed_model_cpkt,
    check_val_losses
    )


class Evaluator():
    def __init__(self, args, logger, model, tokenizer, output_dir_path):
        self.args = args
        self.logger = logger
        self.model = model 
        self.tokenizer = tokenizer
        self.output_dir_path = output_dir_path
        self._set_results_path()

        self.batch_size = self.args.eval_batch_size if self.args.eval_batch_size is not None else self.args.batch_size


    def _set_results_path(self):
        method_name = f"{self.args.task_name}_"
        method_name += self.args.model_name.split("/")[-1] + "_"

        if self.args.checkpoint is not None and self.args.checkpoint == 0:
            method_name += f"from_scratch_"
        elif self.args.checkpoint is not None and self.args.checkpoint != 0:
            method_name += f"checkpoint_{self.args.checkpoint}_"

        method_name += f"{self.args.max_steps}_"

        if self.args.break_steps is not None:
            method_name += f"break_{self.args.break_steps}_"

        if self.args.resume_model_path is not None:
            weight_str, break_steps = parse_resumed_model_cpkt(self.args)
            if self.args.remaining_steps != 100:
                method_name += f"resume_{weight_str}_{break_steps}_remaining_{self.args.remaining_steps}_"
            else:
                method_name += f"resume_{weight_str}_{break_steps}_"


        if self.args.n_epochs != 1:
            method_name += f"epochs_{self.args.n_epochs}_"

            
        if self.args.sample_rule is not None:
            method_name += f"{self.args.sample_rule}_"
        
        slice_list = self.args.slice_list
        if slice_list is not None:
            if len(slice_list) == 1:
                slice_list = slice_list[0]
                if "txt" in slice_list:
                    method_name += f"{slice_list.split('/')[-1]}_"
                else:
                    method_name += f"{slice_list}_"                
            else:
                slice_list = "_".join(slice_list)
                if len(slice_list) > 50:
                    slice_list = slice_list[:50] # truncate 
                method_name += f"{slice_list}_"
            

        
        if self.args.proportions is not None:
            proportions_str = "".join([str(int(i)) if i.is_integer() else str(float(i)) for i in self.args.proportions])
            method_name += f"weights_{proportions_str}_"

        if self.args.proportions_schedule is not None:
            proportions_str = "_".join([str(float(i)) for i in self.args.proportions_schedule])

            if len(proportions_str) > 40:
                proportions_str = proportions_str[:40]
            method_name += f"weightschedule_{proportions_str}_"


                    
        if self.args.graph is not None:
            method_name += "graph_"
        if self.args.graph_path is not None:
            path_str = self.args.graph_path.split("/")[-1].split(".")[0]
            slice_str = "_".join(self.args.slice_list)
            path_str = path_str.split(f"{slice_str}_")[-1]
            path_str = path_str.split("_seed")[0]
            method_name += f"graph_{path_str}_"

            
        if self.args.skillit:
            method_name += f"greedy_{self.args.update_steps}_"
            method_name += f"eta_{self.args.eta}_"
            method_name += f"lookback_{self.args.skillit_window}_"
            
        elif self.args.doge:
            method_name += f"doge_trainer_mu_{self.args.doge_mu}_"

        elif self.args.doremi:
            method_name += f"doremi_trainer_mu_{self.args.doremi_mu}_"

        elif self.args.aioli:
            method_name += f"aioli_eta_{self.args.eta}_update_{self.args.update_steps}_rounds_{self.args.lp_rounds}_steps_{self.args.lp_steps}_"
            if self.args.aioli_prior is not None:
                prior_str = "".join([str(int(i)) if i.is_integer() else str(float(i)) for i in self.args.aioli_prior])
                method_name += f"prior_{prior_str}"

            if self.args.prior_steps is not None:
                method_name += f"{self.args.prior_steps}_"

            if self.args.one_hot_factor != 1:
                method_name += f"ohf_{self.args.one_hot_factor}_"

            if self.args.ema is not None:
                method_name += f"ema_{self.args.ema}_"

            if self.args.aioli_normalize_A:
                method_name += f"normalized_"

            if self.args.aioli_diagonal:
                method_name += f"diagonal_"

        else:
            method_name += "static_"
            if self.args.eta is not None:
                method_name += f"eta_{self.args.eta}_"
            
        if self.args.lr != 5e-5:
            method_name += f"lr_{self.args.lr}_"
        if self.args.lr_scheduler != "linear":
            method_name += f"{self.args.lr_scheduler}_"

        if self.args.doge and self.args.break_steps is not None:
            method_name += f"avg_{self.args.doge_break_step_average}_steps"

        if self.args.custom_name is not None:
            method_name += f"{self.args.custom_name}_"

      
        if method_name.endswith("_"):
            method_name = method_name[:-1]
        
        self.result_path = os.path.join(self.output_dir_path, method_name)
        self.logger.info(f"Output path is {self.result_path}")
        
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)    


    def evaluate(self, tokenized_data, counter, weights):
        """Evaluates the model on a given dataset by computing and saving the loss per sample.
        
        Args: 
        - tokenized_data: a torch dataset to evaluate the model on. 
        - counter: the training step at which the model is evaluated. This is used to help name the results file.
        - weights: if this is not None, we also save the weight per skill at the given training step.
        """
        pass 
            
    def evaluate(self, tokenized_data, counter, weights, split="val", do_not_log=False):
        
        self.model.eval()
        loss_dict = defaultdict(list)


        val_dataloader = self._make_dataloader(tokenized_data)
        for i, data in tqdm(enumerate(val_dataloader)):
            skills = data['skill'] 

            losses = self.val_step(data, i)

            for j, skill in enumerate(skills):
                loss_dict[skill].append(losses[j])   


        result_df = log_val_loss_per_skill(self.logger, loss_dict, counter, split, do_not_log) 


        save_loss(result_df, self.result_path, self.args.selection_seed, counter, split)
        save_weights(weights, self.result_path, self.args.selection_seed, counter, self.args.slice_list, split, do_not_log)

        return result_df 

    
    def _make_dataloader(self, tokenized_data):
        string_columns = ["skill"]
        data_collator = StringDataCollator(self.tokenizer, string_columns, mlm=False)

        dataloader = DataLoader(
            tokenized_data,
            batch_size=self.batch_size,
            sampler=None,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True
        )
        return dataloader
    

    def val_step(self, batch, i):
        input_ids = batch['input_ids'].to('cuda')
        labels = batch['input_ids'].clone().to('cuda')
        labels[labels == self.tokenizer.pad_token_id] = -100 

        if "unmask_span" in batch:
            spans = batch['unmask_span']
            for j in range(len(labels)):
                labels[j, np.arange(spans[j]).astype(int)] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels, output_hidden_states=True, return_dict=True)
            losses = outputs.loss.cpu()
            if i == 0:
                self.context_length = int(len(losses) / self.batch_size)
                
            losses = losses.view(-1, self.context_length)

            keep = losses != 0
            losses = (losses * keep).sum(dim = 1) / keep.sum(dim = 1)

            losses, _ = check_val_losses(losses)


        return losses