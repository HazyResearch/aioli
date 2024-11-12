
from torch.utils.data import DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler


import warnings

from transformers import get_scheduler

from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


import torch

import math


from transformers.utils import ExplicitEnum

import sys
sys.path.append("..")
from evaluator.utils import StringDataCollator

from dataset.slimpj_dataset import SlimpjDataset
from dataset.instruction_dataset import InstructionDataset


def get_train_dataset(args, logger, tokenizer):
    if args.task_name == "slimpj":
        train_dataset = SlimpjDataset(args, logger, tokenizer, args.selection_seed, args.sample_rule, split="train", data_path=args.train_data_dir)
    elif args.task_name == "instruction":
        train_dataset = InstructionDataset(args, logger, tokenizer, args.selection_seed, args.sample_rule, split="train")
    return train_dataset

    
def get_tokenized_train_dataset(args, train_dataset, n_data, include_metadata=False, mapped_train=False):
    if args.task_name == "slimpj":
        tokenized_train = train_dataset.get_tokenized_dataset(n_data, mapped_train=mapped_train)
    elif args.task_name == "instruction":
        tokenized_train = train_dataset.get_tokenized_dataset(n_data, include_metadata)
    return tokenized_train

def get_tokenized_val_dataset(args, validation_dataset):
    if args.task_name in ['slimpj', 'instruction']:
        tokenized_val = validation_dataset.get_tokenized_dataset()
    return tokenized_val


def get_steps(args, n_epochs=None):
    """Computes the number of steps per checkpoint and the total number of training steps."""
    if n_epochs is None:
        n_epochs = args.n_epochs
    ckpt_steps = int(args.max_steps * n_epochs / args.num_ckpts)
    
    total_steps = args.max_steps * n_epochs 

    print(f"Total steps: {total_steps} Steps per checkpoint: {ckpt_steps}")
    
    if args.update_steps is not None:
        assert (total_steps % args.update_steps == 0)    
    
    return ckpt_steps, total_steps

def get_update_steps(args, total_steps):
    """Computes the number of samples per update and the number of total updates (e.g., number of rounds T)."""
    update_size = args.update_steps * args.batch_size 
    n_updates = total_steps / args.update_steps
    return update_size, n_updates


def get_n_data(args):
    if args.skillit or args.proportions_schedule is not None:
        return args.update_steps * args.batch_size
    else:
        return args.max_steps * args.batch_size


def get_train_dataloader(tokenizer, tokenized_dataset, batch_size):
    """
        Returns DataLoader object for training data. 
    """  
    string_columns = ["skill"]
    data_collator = StringDataCollator(tokenizer, string_columns, mlm=False)

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    
def create_optimizer_scheduler(model, lr, max_steps, lr_scheduler_type, end_lr, warmup_steps=50):
    """
        Create AdamW optimizer and learning rate scheduler.
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.00,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    if lr_scheduler_type=="cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, eta_min=end_lr)
    else:
        scheduler = get_scheduler_extended(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps, 
            num_training_steps=max_steps,
        )

    return optimizer, scheduler

def save_model_and_optimizer(model, optimizer, scheduler, model_path):
    print(f"Saving model, optimizer, and scheduler to {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, model_path)

def load_model_and_optimizer(model, optimizer, scheduler, remaining_steps, model_path):
    print(f"Loading model, optimizer, and scheduler from {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if remaining_steps is None:
        raise ValueError("Need to specify how long to train for after resuming training. Please set --remaining_steps.")
    
    return model, optimizer, scheduler, remaining_steps


class LinearWarmupCosineLR(LRScheduler):
    """
    Cosine LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * (1 + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]


class ExtendedSchedulerType(ExplicitEnum):
    LINEAR_WARMUP_COSINE = "linear_warmup_cosine"


# extend scheduler function mapping
TYPE_TO_EXTENDED_SCHEDULER_FUNCTION = {
        ExtendedSchedulerType.LINEAR_WARMUP_COSINE: LinearWarmupCosineLR
}

def get_scheduler_extended(
    name,
    optimizer,
    num_warmup_steps=0,
    num_training_steps=0,
    lr_end=1e-4,
):

    try:
        name = ExtendedSchedulerType(name)
        schedule_func = TYPE_TO_EXTENDED_SCHEDULER_FUNCTION[name]
    except ValueError:
        return get_scheduler(name, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_end=lr_end)



