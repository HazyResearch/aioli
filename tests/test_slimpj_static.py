from argparse import Namespace
from transformers import AutoTokenizer, set_seed

import torch 

from torch.nn.utils import clip_grad_norm_
import numpy as np

import wandb 

from utils import get_val_dataset, get_test_dataset, get_logger, get_model, get_trainer
from trainer.utils import get_train_dataset, get_tokenized_val_dataset, get_tokenized_train_dataset, get_n_data, get_steps, create_optimizer_scheduler, get_train_dataloader
from evaluator.evaluator import Evaluator

from main import get_parser

def test_data():
    output_dir_path = "./tests/"
    logger = get_logger(output_dir_path)


    parser = get_parser()
    args = parser.parse_args([])

    setattr(args, 'task_name', 'slimpj')
    setattr(args, 'slice_list', ['arxiv', 'stackexchange'])
    setattr(args, 'selection_seed', 0)
    setattr(args, 'sample_rule', 'mixture')
    setattr(args, 'proportions', [4.0, 6.0])
    setattr(args, 'model_name', 'EleutherAI/pythia-160m')
    setattr(args, 'context_length', 2048)
    setattr(args, 'max_steps', 5000)
    setattr(args, 'batch_size', 8)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.context_length, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 

    logger.info("Constructing validation data.")
    validation_data = get_val_dataset(args, logger, tokenizer)
    tokenized_val = get_tokenized_val_dataset(args, validation_data)

    assert tokenized_val[0]['skill'] == 'arxiv'
    assert tokenizer.decode(tokenized_val[0]['input_ids'], skip_special_tokens=True)[:50] == '\\section{Introduction}\\label{sec: intro}\nIn early '


def test_static_trainer():
    output_dir_path = "./tests/"
    logger = get_logger(output_dir_path)

    parser = get_parser()
    args = parser.parse_args([])
    setattr(args, 'task_name', 'slimpj')
    setattr(args, 'slice_list', ['arxiv', 'stackexchange'])
    setattr(args, 'selection_seed', 0)
    setattr(args, 'sample_rule', 'mixture')
    setattr(args, 'proportions', [4.0, 6.0])
    setattr(args, 'model_name', 'EleutherAI/pythia-160m')
    setattr(args, 'context_length', 2048)
    setattr(args, 'max_steps', 5000)
    setattr(args, 'batch_size', 8)
    setattr(args, 'checkpoint', 0)
    setattr(args, 'num_ckpts', 500)
    setattr(args, 'lr', 0.0005)
    setattr(args, 'lr_scheduler', 'linear_warmup_cosine')
    setattr(args, 'warmup_steps', 500)
    setattr(args, 'use_flash_attention', True)
    setattr(args, 'do_not_save', True)
    setattr(args, 'break_steps', 11)

    set_seed(args.selection_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.context_length, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    model = get_model(args, logger)

    logger.info("Constructing validation data.")
    validation_data = get_val_dataset(args, logger, tokenizer)

    logger.info("Constructing test data.")
    test_data = get_test_dataset(args, logger, tokenizer)

    logger.info("Constructing training data.")
    train_data = get_train_dataset(args, logger, tokenizer)

    evaluator = Evaluator(args, logger, model, tokenizer, output_dir_path) 

    trainer = get_trainer(
    args, 
    logger=logger, 
    tokenizer=tokenizer, 
    model=model, 
    validation_data=validation_data, 
    test_data=test_data,
    train_data=train_data, 
    evaluator=evaluator) 

    wandb.init(entity="hazy-research", project="data-mixing", mode="disabled")

    loss_all, df = trainer.train()


    assert np.abs(df.task_loss.values[0] - 10.916223) < 0.001
    assert np.abs(df.task_loss.values[1] - 10.868590) < 0.001

    assert loss_all.item() == 10.875