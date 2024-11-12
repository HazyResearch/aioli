"""Utility functions."""
import os
import logging

from typing import Optional, Tuple, Union

import torch 
import numpy as np

import wandb


from transformers import GPTNeoXForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.nn import CrossEntropyLoss



from trainer.skillit_trainer import SkillitTrainer 
from trainer.static_trainer import StaticTrainer
from trainer.doge_trainer import DogeTrainer
from trainer.doremi_trainer import DoremiTrainer
from trainer.aioli_trainer import AioliTrainer
from trainer.utils import get_tokenized_val_dataset, get_steps

from dataset.slimpj_dataset import SlimpjDataset
from dataset.instruction_dataset import InstructionDataset



def make_output_dir(output_dir: str, run_id: str) -> str:
    dir_path = os.path.join(output_dir, run_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def get_logger(dir_path):
    # Create a logger
    logger = logging.getLogger("Logging")
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to output.log
    file_handler = logging.FileHandler(os.path.join(dir_path, "output.log"))
    file_handler.setLevel(logging.INFO)

    # Create a stream handler that prints to the screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.propagate = False

    return logger

def get_model_type(model_name):
    model_map = {
        "pythia": GPTNeoXForCausalLMLossPerPoint,
    } 

    model_class = [v for k, v in model_map.items() if k in model_name]

    if len(model_class) != 1:
        raise ValueError(f"Either unknown model name {model_name} or incorrect model_map.")
    return model_class[0]

def load_model(model_name, model_path):
    model_class = get_model_type(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = model_class(config)
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    return model 

def get_model(args, logger):
    model_class = get_model_type(args.model_name)

    if args.checkpoint is not None and args.checkpoint != 0:
        valid_pythia_checkpoints = np.concatenate([np.array([0]), 2**np.arange(10), np.arange(1,144)*1000])
        assert args.checkpoint in valid_pythia_checkpoints, f"Checkpoint {args.checkpoint} is not a valid Pythia checkpoint!"

        logger.info(f"Training {args.model_name} from checkpoint {args.checkpoint}!")
        if args.use_flash_attention:
            logger.info("Using flash attention and bf16.")
            model = model_class.from_pretrained(args.model_name, revision="step"+str(args.checkpoint), 
                                                attn_implementation="flash_attention_2", 
                                                torch_dtype=torch.bfloat16, 
                                                trust_remote_code=True)
        else:
            model = model_class.from_pretrained(args.model_name, revision="step"+str(args.checkpoint), trust_remote_code=True)
    elif args.checkpoint == 0:
        logger.info(f"Training {args.model_name} from scratch!")
        if args.use_flash_attention:
            logger.info("Using flash attention and bf16.")
            config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True,
                                                attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)  
        else:
            config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)  

        if args.use_flash_attention:
            model = model_class._from_config(config, torch_dtype=torch.bfloat16)

            first_param = next(model.parameters())
            model_dtype = first_param.dtype

            if model_dtype == torch.bfloat16:
                print("Model is using torch.bfloat16")
            else:
                print(f"Model is using data type: {model_dtype}")

        else:
            model = model_class(config)
    else:
        logger.info(f"Continually training {args.model_name}!")

        if args.use_flash_attention:
            logger.info("Using flash attention and bf16.")
            model = model_class.from_pretrained(args.model_name, trust_remote_code=True,
                                           attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)       
        else:
            model = model_class.from_pretrained(args.model_name, trust_remote_code=True,
                                           )       

    return model.cuda()

def eval_only(args, model_path):
    if args.use_saved_model and os.path.exists(model_path):
        return True
    return False 

def evaluate_existing_model(args, logger, evaluator, tokenizer, run_name):
    validation_data = get_val_dataset(args, logger, tokenizer)
    tokenized_val = get_tokenized_val_dataset(args, validation_data)
    _, total_steps = get_steps(args)
    wandb.init(entity="mayeechen", project="skill-it", name=run_name)
    evaluator.evaluate(tokenized_val, total_steps, None)
 
def get_trainer(args, **kwargs):
    if args.skillit:
        trainer = SkillitTrainer(args, **kwargs)
    elif args.doremi:
        trainer = DoremiTrainer(args, **kwargs)
    elif args.doge:
        if args.task_name not in ['slimpj']:
            raise NotImplementedError("We do not have the mix skill implemented for other datasets yet.")
        trainer = DogeTrainer(args, **kwargs)
    elif args.aioli:
        trainer = AioliTrainer(args, **kwargs)
    else:
        trainer = StaticTrainer(args, **kwargs)
    return trainer 

def get_val_dataset(args, logger, tokenizer):
    if args.task_name == "slimpj":
        seed = 42
        val_data = SlimpjDataset(args, logger, tokenizer, seed, sample_rule="stratified", split="val", data_path=args.val_data_dir)
    elif args.task_name == "instruction":
        seed = 42
        val_data = InstructionDataset(args, logger, tokenizer, seed, sample_rule="stratified", split="val")
    else:
        raise NotImplementedError(f"Unknown task {args.task_name}")
    return val_data

def get_test_dataset(args, logger, tokenizer):
    if args.task_name == "slimpj":
        seed = 42
        val_data = SlimpjDataset(args, logger, tokenizer, seed, sample_rule="stratified", split="test", data_path=args.val_data_dir)
    elif args.task_name == "instruction":
        seed = 42
        val_data = InstructionDataset(args, logger, tokenizer, seed, sample_rule="stratified", split="test")
    else:
        raise NotImplementedError(f"Unknown task {args.task_name}")
    return val_data



class GPTNeoXForCausalLMLossPerPoint(GPTNeoXForCausalLM):
    """
        GPTNeoXForCausalLM with `CrossEntropyLoss(reduction=none)` in `forward()` to obtain per-sample losses when evaluating. 
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_per_sample_loss: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config.is_decoder = True
        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()


        if return_per_sample_loss:
            loss_fct = CrossEntropyLoss(reduction="none")
        else:
            loss_fct = CrossEntropyLoss(reduction="mean")

        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))


        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
            