import numpy as np
import pickle
import os
import json
import torch

from .dataset import AbstractDataset


TASK_TO_FULL = {
    'squad': 'task075_squad1.1_answer_generation',
    'mathqa': 'task1420_mathqa_general',
    'xsum': 'task1290_xsum_summarization',
    'sst2': 'task363_sst2_polarity_classification',
    'abductivenli': 'task067_abductivenli_answer_generation',
    'hellaswag': 'task1389_hellaswag_completion',
    'semeval': 'task295_semeval_2020_task4_commonsense_reasoning',
    'piqa': 'task080_piqa_answer_generation',
    'boolq': 'task380_boolq_yes_no_question'
}

def make_split():
    task_dir = "../../natural-instructions/tasks/"

    np.random.seed(0) 

    all_train_idxs = {}
    all_test_idxs = {}
    all_val_idxs ={}
    
    selected_tasks = list(TASK_TO_FULL.values())

    for i, task in enumerate(selected_tasks):
        path = os.path.join(task_dir, task + ".json")
        with open(path, "r") as f:
            data = json.load(f)

        n = len(data['Instances'])

        shuffled = np.random.permutation(np.arange(n))
        val_idxs = shuffled[:100].tolist()
        test_idxs = shuffled[100:200].tolist()
        train_idxs = shuffled[200:].tolist()

        all_val_idxs[task] = val_idxs

        all_test_idxs[task] = test_idxs

        all_train_idxs[task] = train_idxs


    with open("./aux_data/instruction_train_split.pkl", "wb") as f:
        pickle.dump(all_train_idxs, f)

    with open("./aux_data/instruction_val_split.pkl", "wb") as f:
        pickle.dump(all_val_idxs, f)

    with open("./aux_data/instruction_test_split.pkl", "wb") as f:
        pickle.dump(all_test_idxs, f)



class InstructionDataset(AbstractDataset):
    """
        Dataset that contains mixture over instruction tuning tasks.
    """

    def __init__(
        self,
        args,
        logger,
        tokenizer,
        seed,
        sample_rule, 
        split,
    ):
        super().__init__(args, logger, tokenizer, seed, sample_rule, split)

        self.k = args.k 
        proportions = args.proportions
        self.set_skills(args)
        self.set_proportions(args, proportions)

    def set_skills(self, args):
        slices = args.slice_list
        if slices is not None:
            self.skills = np.array(slices, dtype=object)
        else:
            self.skills = sorted(list(TASK_TO_FULL.keys()))
        self.k = len(self.skills)
        self.n_var = len(self.skills)
        self.logger.info(f"{self.k} skills:\n{self.skills}")

    def set_proportions(self, args, proportions):
        if self.sample_rule == "mixture":
            if proportions is not None:
                proportions = np.array(proportions)
                proportions /= sum(proportions)
                self.proportions = proportions 

                assert len(self.proportions) == self.k, f"Length of proportions is {len(self.proportions)} but k is {self.k}"
                self.logger.info(f"Setting skill proportions:\n{self.proportions}")
            else:
                self.logger.warning("Sample rule is mixture but proportions is not specified.")
        elif self.sample_rule == "stratified":
            self.proportions = np.repeat(1.0 / self.n_var, self.n_var) 
        else:
            raise ValueError("Instruction dataset does not support iid per-sample random sampling yet")
        

    def get_tokenized_dataset(self, n_data=None, include_metadata=False):
        if self.is_eval:
            return self.get_tokenized_val()
        else:
            return self.get_tokenized_train(n_data, include_metadata)


    def get_tokenized_val(self):
        self.logger.info(f"Getting {self.split} data.")
        tokenized_data = []
        for i, skill in enumerate(self.skills):
            skill_dataset = InstructionTaskDataset(self.args, self.logger, self.tokenizer, skill, None, self.split, 
                self.rng,
            )
            tokenized_data.extend(skill_dataset.dataset)

        dataset = InstructionTuningTorchDataset(tokenized_data, self.is_eval)
        return dataset



    def get_tokenized_train(self, n_data, include_metadata):
        n_per_skill = (n_data * self.proportions).astype(int)
        n_per_skill[-1] = n_data - n_per_skill[:-1].sum()

        self.logger.info(f"Getting training data. Probabilities: {list(zip(self.skills, self.proportions))}")

        tokenized_data = []
        for i, (skill, n) in enumerate(zip(self.skills, n_per_skill)):
            skill_dataset = InstructionTaskDataset(self.args, self.logger, self.tokenizer, skill, n, self.split, 
                self.rng,
            )
            tokenized_data.extend(skill_dataset.dataset)

        order = self.rng.permutation(len(tokenized_data))
        tokenized_data = [tokenized_data[order[i]] for i in range(len(tokenized_data))]
        
        dataset = InstructionTuningTorchDataset(tokenized_data, self.is_eval, include_metadata)

        return dataset



class InstructionTuningTorchDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, is_eval, include_metadata=False):
        self.data = tokenized_data
        self.is_eval = is_eval
        self.include_metadata = include_metadata

        
    def __getitem__(self, idx):
        data = self.data[idx]
        
        if self.is_eval:
            return {
                "input_ids": data['input_ids'],
                "attention_mask": data['attention_mask'],
                "skill": data['skill'],
                "unmask_span": data['unmask_span']
            }
        else:
            if self.include_metadata:
                return {
                    "input_ids": data['input_ids'],
                    "attention_mask": data['attention_mask'],
                    "skill": data['skill'],
                    "unmask_span": data['unmask_span']
                }
            else:
                return {
                    "input_ids": data['input_ids'],
                    "attention_mask": data['attention_mask'],
                    "unmask_span": data['unmask_span'],
                    "skill": data['skill'],
                }
            
    def __len__(self):
        return len(self.data)





class InstructionTaskDataset():
    def __init__(self, args, logger, tokenizer, task_name, n_data, split, rng):
        self.args = args
        self.logger = logger 
        self.tokenizer = tokenizer 
        self.task_name = task_name
        self.n_data = n_data 
        self.split = split 
        self.is_eval = self.split in ['test', 'val']
        self.rng = rng

        self.dataset = []

        self.generate_dataset()


    def create_task_data(self):
        self.logger.info(f"Tokenizing data from {self.task_name} {self.split} split for the first time.")

        with open(f"./aux_data/instruction_{self.split}_split.pkl", "rb") as f:
            split_idxs = pickle.load(f)[TASK_TO_FULL[self.task_name]]
        data_path = os.path.join("../natural-instructions/tasks", TASK_TO_FULL[self.task_name] + ".json")
        with open(data_path, 'r') as f:
            data = json.load(f)
        samples = [instance for i, instance in enumerate(data['Instances']) if i in split_idxs]

        definition = data['Definition'][0]
        formatted_samples = []
        for sample in samples:
            prompt_no_answer = definition + "Input: " + sample['input'] + "\n"
            answer = self.rng.choice(sample['output'])
            full_text = prompt_no_answer + answer

            tokenized_text = self.tokenizer(full_text.strip(), padding="max_length")

            if len(tokenized_text['input_ids']) > self.args.context_length:
                self.logger.warning(f"Task name: {self.task_name}. Context out of bounds: {len(tokenized_text['input_ids'])} is greater than {self.args.context_length}.")
                continue 
            
            
            unmask_span = len(self.tokenizer(prompt_no_answer)['input_ids'])
            offset = 0 
            while True:
                if answer.startswith(self.tokenizer.decode(tokenized_text['input_ids'][unmask_span - offset]).strip()):
                    break
                offset += 1 
            unmask_span -= offset
    
            entry = {
                'input_ids': torch.LongTensor(tokenized_text['input_ids']),
                'attention_mask': torch.LongTensor(tokenized_text['attention_mask']),
                'unmask_span': unmask_span,
                'skill': self.task_name
            }
        
            formatted_samples.append(entry)

        with open(os.path.join(self.args.train_data_dir, f"{self.task_name}_{self.split}.pkl"), "wb") as f:
            pickle.dump(formatted_samples, f)

        return formatted_samples 

    def generate_dataset(self):
        data_file = os.path.join(self.args.train_data_dir, f"{self.task_name}_{self.split}.pkl")
        if os.path.exists(data_file):
            with open(data_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = self.create_task_data()

        # depending on split, need to subsample 
        if self.split == "train":
            n_train = len(samples)
            assert self.n_data <= n_train, f"number of samples requested needs to be less than {n_train} --- otherwise repeated data"
            n_data_idxs = self.rng.choice(np.arange(n_train), size=self.n_data, replace=False)
            samples = [sample for i, sample in enumerate(samples) if i in n_data_idxs]
        
        self.dataset = samples
