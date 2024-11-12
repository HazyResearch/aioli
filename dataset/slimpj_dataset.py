"""
Code adapted from https://github.com/Olivia-fsm/DoGE.
"""
import numpy as np
import torch
import os
from transformers import AutoTokenizer
from datasets import load_dataset



from .dataset import AbstractDataset

from datasets import Dataset, IterableDataset, concatenate_datasets
from typing import Iterator
from datasets.iterable_dataset import RandomlyCyclingMultiSourcesExamplesIterable
from copy import deepcopy

from tqdm import tqdm

import tiktoken
tknzr = tiktoken.get_encoding("gpt2")


SUBSET2META = {
    'arxiv': 'RedPajamaArXiv',
    'book': 'RedPajamaBook',
    'cc': 'RedPajamaCommonCrawl',
    'c4': 'RedPajamaC4',
    'github': 'RedPajamaGithub',
    'stackexchange': 'RedPajamaStackExchange',
    'wikipedia': 'RedPajamaWikipedia',
    
}


class _HasNextIterator(Iterator):
    """Iterator with an hasnext() function. Taken from https://stackoverflow.com/questions/1966591/has-next-in-python-iterators."""

    def __init__(self, it):
        self.it = iter(it)
        self._hasnext = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._hasnext:
            result = self._thenext
        else:
            result = next(self.it)
        self._hasnext = None
        return result

    def hasnext(self):
        if self._hasnext is None:
            try:
                self._thenext = next(self.it)
            except StopIteration:
                self._hasnext = False
            else:
                self._hasnext = True
        return self._hasnext

def domain_gen(data, seq_len, skill=None):
    if skill is None:
        for i in range(len(data)//seq_len):
            yield {"input_ids": data[i*seq_len:(i+1)*seq_len]}
    else:
        for i in range(len(data)//seq_len):
            yield {"skill": skill, "input_ids": data[i*seq_len:(i+1)*seq_len]}


def df_gen(data, domain_id=None):
    for i in range(len(data)):
        yield {"skill": torch.tensor([domain_id], dtype=torch.long), "input_ids": torch.LongTensor(data.tokenized.values[i])}


class UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        RandomlyCyclingMultiSourcesExamplesIterable):

    def __init__(self, ex_iterables, generator, is_eval, probabilities=None, probabilities_handle=None, stopping_strategy="all_exhausted",
                 curriculum_dict=None, random_bsz=64):
        '''
        probabilities: vector of static probabilities over training
        probabilities_handle: handle to domain weights buffer in model params
        '''
        super().__init__(ex_iterables, generator, stopping_strategy=stopping_strategy)

        self.stopping_strategy = ""

        self.probabilities_handle = probabilities_handle
        self.probabilities = probabilities
        self.curriculum_dict = curriculum_dict

        self.is_eval = is_eval
        if curriculum_dict is not None:
            self.step = 0

        self.random_bsz = random_bsz
            
    @staticmethod
    def _iter_random_indices(rng, num_sources, probabilities_handle=None, p=None, random_batch_size=64):
        while True:
            if probabilities_handle is not None:
                p = probabilities_handle.detach().cpu().numpy()

            p /= sum(p)

            yield from (int(i) for i in rng.choice(len(p), random_batch_size, p=p))

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        return self._iter_random_indices(rng, len(self.ex_iterables), probabilities_handle=self.probabilities_handle, p=self.probabilities, random_batch_size=self.random_bsz)

    def shard_data_sources(self, shard_indices):
        return self

    @property
    def n_shards(self):
        return 1

    def shuffle_data_sources(self, seed):
        self.ex_iterables = [ex_iterable.shuffle_data_sources(seed) for ex_iterable in self.ex_iterables]
        return self
    
    def __iter__(self):
        iterators = [_HasNextIterator(ex_iterable) for ex_iterable in self.ex_iterables]

        indices_iterator = self._get_indices_iterator()

        is_exhausted = np.full(len(self.ex_iterables), False)
        for i in indices_iterator:
            try:  # let's pick one example from the iterator at index i
                yield next(iterators[i])

                # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                if not iterators[i].hasnext():
                    is_exhausted[i] = True


                    if self.is_eval:
                        # we do not oversample eval datasets ever! 
                        self.probabilities[i] = 0
                        self.probabilities /= self.probabilities.sum()
                        if self.probabilities_handle is not None:
                            self.probabilities_handle[i] = 0
                            self.probabilities_handle /= self.probabilities_handle.sum()

                    nonzero_idxs = torch.nonzero(self.probabilities)
                    
                    
                    if self.bool_strategy_func(is_exhausted[nonzero_idxs]):
                        if self.is_eval:
                            break

                    # otherwise reinitialise the iterator and yield the first example
                    iterators[i] = _HasNextIterator(self.ex_iterables[i])

            except StopIteration:
                # here it means that the i-th iterabledataset is empty, i.e we never have the occasion to yield an element of the i-th dataset.
                # we still check if the stopping criteria is met and if we break out of the loop in case of an oversampling strategy
                is_exhausted[i] = True

                if self.bool_strategy_func(is_exhausted):
                    # if the stopping criteria is met, break the main for loop
                    break

def interleave_datasets(datasets, is_eval, batch_size, probabilities=None, probabilities_handle=None, seed=None, stopping_strategy='all_exhausted'):
    iterable_datasets = []

    for dataset in datasets:
        if not isinstance(dataset, IterableDataset):
            iterable_datasets.append(dataset.to_iterable_dataset())
        else:
            iterable_datasets.append(dataset)

    ex_iterables = [d._ex_iterable for d in iterable_datasets]

    generator = np.random.default_rng(seed)

    ex_iterable = UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, generator=generator, is_eval=is_eval,
            probabilities=probabilities, probabilities_handle=probabilities_handle,
            stopping_strategy=stopping_strategy,
            random_bsz=batch_size,
            )

    return IterableDataset(ex_iterable=ex_iterable)


def interleave_per_batch_mapped(datasets, proportions, batch_size, rng):
    # Calculate samples per batch for each dataset
    samples_per_batch = np.array([max(1, int(ratio * batch_size)) for ratio in proportions])

    samples_per_batch[-1] = batch_size - samples_per_batch[:-1].sum()
    
    # Create iterators
    iterators = [iter(dataset) for dataset in datasets]

    interleaved_data = []
    
    while True:
        batch = []
        try:
            for iterator, samples in zip(iterators, samples_per_batch):
                batch.extend([next(iterator) for _ in range(samples)])
        except StopIteration:
            break
        
        rng.shuffle(batch)
        interleaved_data.extend(batch)

    return Dataset.from_list(interleaved_data)

def get_slimpajama_6b(data_path, subset='arxiv', num_proc=40,
                          return_torch=False, tokenizer_name="EleutherAI/pythia-160m"):
    """ Full: https://huggingface.co/datasets/cerebras/SlimPajama-627B
        6B-subset: DKYoon/SlimPajama-6B
    """
    subset_name = SUBSET2META[subset]
    print('Load subset_name: ', subset_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token 
    REDPAJAMA_DATA_PATH = data_path
    
    SUBSET_PATH = os.path.join(REDPAJAMA_DATA_PATH, subset)
    if not os.path.exists(os.path.join(SUBSET_PATH, 'test.bin')):
        os.makedirs(SUBSET_PATH, exist_ok=True)
        dataset = load_dataset("DKYoon/SlimPajama-6B", split=['train', 'test'])
        print(dataset)
        data_dict = {}
        data_dict['train'] = dataset[0].filter(lambda example: example["meta"]['redpajama_set_name']==subset_name)

        # split test set into val/test by shuffling; then evens=val, odds=test.
        val_test = dataset[1].filter(lambda example: example["meta"]['redpajama_set_name']==subset_name)
        val_test = val_test.shuffle(seed=42)
        data_dict['val'] = val_test.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
        data_dict['test'] = val_test.filter(lambda example, idx: idx % 2 == 1, with_indices=True)

        def process_hf_tokenizer(example):
            'Processing dataset...'
            ids = tokenizer(example['text'])['input_ids'] # for gptneoxtokenizer, no max tok length
            out = {'ids': ids, 'len': len(ids)}
            return out
        
        # tokenize the dataset
        tokenized = {}


        tokenized['train'] = data_dict['train'].map(
            process_hf_tokenizer,
            remove_columns=['text', 'meta', '__index_level_0__'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        tokenized['val'] = data_dict['val'].map(
            process_hf_tokenizer,
            remove_columns=['text', 'meta', '__index_level_0__'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        tokenized['test'] = data_dict['test'].map(
            process_hf_tokenizer,
            remove_columns=['text', 'meta', '__index_level_0__'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        print(tokenized['val'])
        print(tokenized['test'])

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            print('Columns: ', dset.features)
            arr_len = np.sum(dset['len'])
            filename = os.path.join(SUBSET_PATH, f'{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 10 if len(dset) >= 10 else 1
        
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                print(total_batches, batch_idx)
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(os.path.join(SUBSET_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(SUBSET_PATH, 'val.bin'), dtype=np.uint16, mode='r')
    test_data = np.memmap(os.path.join(SUBSET_PATH, 'test.bin'), dtype=np.uint16, mode='r')

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.int32))
        val_data = torch.tensor(np.array(val_data, dtype=np.int32))
        test_data = torch.tensor(np.array(test_data, dtype=np.int32))

    return {'train': train_data, 'val': val_data, 'test': test_data}



class SlimpjDataset(AbstractDataset):
    def __init__(
        self,
        args,
        logger,
        tokenizer,
        seed,
        sample_rule,
        split,
        data_path,
    ):
        super().__init__(
            args, logger, tokenizer, seed, sample_rule, split, data_path
        )

        self.k = args.k 

        self.set_skills(args)
        self.set_proportions(args, args.proportions)

    def set_skills(self, args):
        slices = args.slice_list

        if slices is None:
            self.skills = np.array(list(SUBSET2META.keys()))
        else:
            self.skills = np.array(slices) 
                
        self.k = len(self.skills)
        self.n_var = len(self.skills)

        self.logger.info(f"{self.k} skills:\n{self.skills}")

    def set_proportions(self, args, proportions):
        if self.sample_rule == "mixture":
            if proportions is not None:
                proportions = np.array(proportions)
                proportions = proportions / sum(proportions)
                self.proportions = proportions 

                assert len(self.proportions) == self.k, f"Length of proportions is {len(self.proportions)} but k is {self.k}"
                self.logger.info(f"Setting skill proportions:\n{self.proportions}")
            else:
                self.logger.warning("Sample rule is mixture but proportions is not specified.")
        elif self.sample_rule == "stratified":
            self.proportions = np.repeat(1.0 / self.n_var, self.n_var)
            self.logger.info(f"Setting skill proportions:\n{self.proportions}")
        else:            
            raise ValueError("Not supported.")
                

    def get_tokenized_dataset(self, n_data=None, mapped_train=False):
        rst_dict = {}

        if mapped_train:
            assert n_data is not None
            assert self.split == "train"
            n_per_skill = (self.proportions * n_data).astype(int)
            n_per_skill[-1] = n_data - n_per_skill[:-1].sum()

        for i, skill in enumerate(self.skills):

            if isinstance(self.data_path, list):
                SUBSET_PATH = os.path.join(self.data_path[0], skill)
            else:
                SUBSET_PATH = os.path.join(self.data_path, skill)

            split_path = os.path.join(SUBSET_PATH, f'{self.split}.bin')
            if not os.path.exists(split_path):
                data = get_slimpajama_6b(self.args.train_data_dir, skill, tokenizer_name="EleutherAI/pythia-160m")
                data = data[self.split]
            else:
                data = np.memmap(os.path.join(SUBSET_PATH, f'{self.split}.bin'), dtype=np.uint16, mode='r')
                

            if mapped_train:
                shuffled_idxs = np.arange(0, len(data), self.args.context_length)
                subselected_idxs = self.rng.choice(shuffled_idxs, size=n_per_skill[i], replace=False)
                data = np.concatenate([data[i:i+self.args.context_length] for i in subselected_idxs])
            elif self.split == "train":
                # Only shuffle the training data, otherwise documents are chunked inconsistently.
                shuffled_idxs = np.arange(0, len(data), self.args.context_length)
                self.rng.shuffle(shuffled_idxs)
                data = np.concatenate([data[i:i+self.args.context_length] for i in shuffled_idxs])

            rst_dict[skill] = data

        data_dict = {dom:v for dom,v in rst_dict.items()}

        dataset_ls = []

        for k in data_dict.keys():
            if self.is_eval or mapped_train:
                domain_dataset = Dataset.from_generator(domain_gen,
                                        gen_kwargs={'data': data_dict[k],
                                                    'seq_len': self.args.context_length,
                                                    'skill': k,
                                                    }
                                        )
                # WARNING: Huggingface caches this dataset! If you make changes to domain_gen, delete the dataset folders (in cache_dir/datasets/generator/*)  
            else:
                domain_dataset = IterableDataset.from_generator(domain_gen,
                                                    gen_kwargs={'data': data_dict[k],
                                                                'seq_len': self.args.context_length,
                                                                'skill': k,
                                                                }
                                                    )
                
            dataset_ls.append(domain_dataset)
        


        if self.is_eval: 
            ds = concatenate_datasets(dataset_ls)
        elif mapped_train:
            ds = interleave_per_batch_mapped(dataset_ls, self.proportions, self.args.batch_size, self.rng)
        else:
            ds = interleave_datasets(
                            dataset_ls,
                            is_eval=self.is_eval,
                            batch_size=self.args.batch_size,
                            probabilities=torch.tensor(self.proportions),
                            probabilities_handle=torch.tensor(self.proportions),
                            seed=self.seed)
        return ds

