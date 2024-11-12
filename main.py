import os
from transformers import AutoTokenizer
import wandb
from transformers import set_seed
import argparse
from datetime import datetime
import torch



from utils import (get_trainer, get_logger, get_val_dataset, get_test_dataset, 
                   make_output_dir, get_model, load_model,
                   eval_only, evaluate_existing_model)

from trainer.utils import get_train_dataset

from evaluator.evaluator import Evaluator

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Aioli data mixing"
    )
    # data arguments
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the dataset. Currently only supports `slimpj` and `instruction.`"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        help="Directory from which to load training data",
    )
    parser.add_argument(
        "--val_data_dir",
        nargs="+",
        type=str,
        help="Directory from which to load validation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Directory where all results are stored."
    )

    parser.add_argument(
        "--selection_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--slice_list",
        type=str,
        nargs="+",
        help="the direct list of skills to use.",
        default=None,
    )
    parser.add_argument(
        "--sample_rule",
        type=str,
        default=None,
        help="Strategy for mixing. `stratified` means 1/k probability per skill, while `mixture` enables using a custom list of proportions.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of skills. Needs to be set for synthetics, but for real datasets this can be inferred from the data.",
    )
    parser.add_argument(
        "--proportions",
        type=float,
        nargs="+",
        default=None,
        help="List of proportions to sample with (static). Does not need to add up to 1. The length should be equal to slice_list if slice_list is set, otherwise k * n_segment.",
    )
    parser.add_argument(
        "--proportions_schedule",
        type=float,
        nargs="+",
        default=None,
        help="List of proportions to sample with. Does not need to add up to 1. If this is set, the training procedure will be divided into len(proportions_schedule)/(len(slice_list) or k) segments of equal length.",
    )

    # training/eval arguments
    parser.add_argument(
        "--context_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--n_epochs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--lr",
        default=5e-5,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of steps to train for. Overrides n_epochs.",
    )
    parser.add_argument(
        "--model_name",
        default="EleutherAI/pythia-160M",
        type=str,
        help="Model config.",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--num_ckpts",
        help="Number of checkpoints to evaluate the model at.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Pretrained model checkpoint. If 0, means we start from scratch. If None, we use the fully pre-trained model."
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
    )

    parser.add_argument(
        "--custom_name",
        default=None,
        type=str,
        help="If we want to add a custom name for the run."
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="If set, saves the model."
    )

    parser.add_argument(
        "--use_saved_model",
        action="store_true",
        help="Checks if there is already a model trained according to the given arguments, and evaluates using this saved model."
    )

    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="use flash attention 2."
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50
    )

    # Arguments for analyzing linear dynamic mixing law
    parser.add_argument(
        "--break_steps",
        default=None,
        type=int,
        help="If set to a value, we will pause training and save the model and optimizer state after this amount of steps"
    )

    parser.add_argument(
        "--do_not_save",
        action="store_true",
        help="If set to true, we do not save the model after break_steps"
    )

    parser.add_argument(
        "--resume_model_path",
        default=None,
        type=str,
        help="If set to a value, we will resume training at a saved checkpoint."
    )

    parser.add_argument(
        "--remaining_steps",
        default=None,
        type=int,
        help="If set to a value, we will resume training at a saved checkpoint for this amount of steps. Usually can be small to test a `local` change."
    )

    parser.add_argument(
        "--end_lr",
        type=float,
        default=0.0,
        help="If set, is the learning rate by the end of training. Used for experiments where we do not train from scratch."
    )

    # Skill-It algorithm arguments
    parser.add_argument(
        "--skillit",
        action="store_true",
        help="Use skillit algorithm"
    )
    parser.add_argument(
        "--update_steps",
        type=int,
        default=None,
        help="How often to update multiplicative weights"
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="eta parameter for weight update"
    )

    parser.add_argument(
        "--skillit_window",
        type=int,
        default=3,
        help="Look-back window for weight update in Skill-It."
    )

    parser.add_argument(
        "--graph_path",
        type=str,
        default=None,
        help="Path to .npy file containing skills graph for Skill-It."
    )

    # DoGE arguments
    parser.add_argument(
        "--doge",
        action="store_true",
        help="If set to true, performs doge proxy run."
    )

    parser.add_argument(
        "--doge_mu",
        default=0.01,
        type=float,
        help="Doge mu hyperparameter for how much to update the score"
    )

    parser.add_argument(
        "--doge_val_batch_size",
        type=int,
        help="The batch size of the val dataset when using it to compute gradients. --batch_size + --doge_val_batch_size should be your overall desired training batch size."
    )

    parser.add_argument(
        "--doge_break_step_average",
        type=int,
        default=5,
        help="When obtaining DoGE's A^t for analysis, this is the number of steps before the break step to start computing the average doge matrix."
    )

    # DoReMi arguments
    parser.add_argument(
        "--doremi",
        action="store_true",
        help="Flag for using doremi trainer"
    )

    parser.add_argument(
        "--doremi_reference_model_path",
        type=str,
        default=None,
        help="Path to reference model for DoReMi (stratified sampling model)"
    )

    parser.add_argument(
        "--doremi_mu",
        type=float,
        default=1.0,
        help="Hyperparameter mu for doremi"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )

    # Aioli arguments
    parser.add_argument(
        "--aioli",
        action="store_true",
        help="Aioli algorithm."
    )

    parser.add_argument(
        "--aioli_normalize_A",
        action="store_true",
        help="If set to true, we normalize the A matrix before using it (to overcome over-emphasizing early-on weights)"
    )

    parser.add_argument(
        "--aioli_prior",
        type=float,
        nargs="+",
        default=None,
        help="Sampling prior throughout the run"
    )

    parser.add_argument(
        "--prior_steps",
        type=int,
        default=None,
        help="If --aioli_prior is set, we need to specify how many steps the static mixture is trained for."
    )

    parser.add_argument(
        "--lp_rounds",
        type=int,
        default=1,
        help="Number of sweeps through the k datasets in the LearnParams subroutine of Aioli."
    )
    parser.add_argument(
        "--lp_steps",
        type=int,
        default=5,
        help="Number of steps to train on each dataset in the LearnParams subroutine of Aioli."
    )

    parser.add_argument(
        "--one_hot_factor",
        type=float,
        default=1,
        help="If set, we use a smoothed vector [one_hot_factor, 1 - one_hot_factor/k, ....] instead of the one-hot vector, and we solve a linear system."
    )

    parser.add_argument(
        "--ema",
        type=float,
        default=None,
        help="exponential moving average parameter for Aioli"
    )

    parser.add_argument(
        "--aioli_diagonal",
        action="store_true",
        help="If true, use diagonal A^t matrix in Aioli. Need to have ohf_factor != 1."
    )

    # Aioli+Skillit arguments
    parser.add_argument(
        "--skillit_eta",
        type=float,
        help="when doing aioli initialized with skillit, this is what we use to control the initial skillit weight."
    )

    parser.add_argument(
        "--skillit_update_steps",
        type=int,
        help="when doing aioli initialized with skillit, this is what we use to control the number of update steps for skill-it."

    )

    return parser

def main():
    run_id = datetime.now().strftime("%m%d%Y")

    parser = get_parser()
    args = parser.parse_args()

    set_seed(args.selection_seed)

    output_dir_path = make_output_dir(args.output_dir, run_id)
    
    logger = get_logger(output_dir_path)
    logger.info(args)

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

    run_name = evaluator.result_path.split("/")[-1]
    model_path = f"saved_models/{run_name}_seed_{args.selection_seed}.pt"
    wandb_name = f"{run_name}_seed_{args.selection_seed}"


    if eval_only(args, model_path):
        model = load_model(args.model_name, model_path)

        logger.info("Model exists. Computing final generations.")
        evaluator.model = model.to('cuda')
        evaluator.result_path = os.path.join(output_dir_path, run_name)
        evaluate_existing_model(args, logger, evaluator, tokenizer, run_name)
    else: 
        wandb.init(entity="hazy-research", project="data-mixing", name=wandb_name)
        model = trainer.train()
        wandb.finish()
    
    if args.save_model:
        logger.info(f"Saving model to {model_path}.")
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
