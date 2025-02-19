import numpy as np
import random 
import torch 


class AbstractDataset():
    """
    The AbstractDataset class is a dataset that can be filtered and sampled from.
    """

    def __init__(
        self, args, logger, tokenizer, seed, sample_rule, split, data_path=None
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
        self.context_length = args.context_length
        self.sample_rule = sample_rule
        self.split = split
        self.is_eval = self.split in ['test', 'val']
        self.data_path = data_path
        self.seed = seed

        self.rng = np.random.default_rng(seed=self.seed)
        

    def set_skills(self, args):
        """Sets the support of skills over which we are sampling from by processing args.slice_list."""
        pass

    def set_proportions(self, args, proportions):
        """Sets the proportions with which to sample each skill.
        
        Arguments:
        - args: arguments
        - proportions: a list of values (not necessarily adding up to 1) that determine how frequently to sample each skill. This is used to update the skills mixture before and during training.
        """
        pass

    def get_tokenized_dataset(self, n_data):
        """Produce a train/test/val dataset of size n_data."""
        pass