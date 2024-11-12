class AbstractTrainer():
    
    def __init__(self, args, logger, tokenizer, model, validation_data, test_data, train_data, evaluator):
        self.args = args
        self.logger = logger 
        self.tokenizer = tokenizer 
        self.model = model 
        self.validation_data = validation_data
        self.test_data = test_data
        self.train_data = train_data 
        self.evaluator = evaluator 

        self.run_name = self.evaluator.result_path.split("/")[-1]

     
    def train(self):
        raise NotImplementedError
    
