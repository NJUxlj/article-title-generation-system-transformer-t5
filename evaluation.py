import torch
import collections
import sys
import json


from transformer.Translator import Translator

from loader import load_data, DataGenerator

from torch.utils.data import DataLoader, Dataset, OrderedDict



class Evaluator(object):    
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data:DataLoader = load_data()
        self.reverse_vocab = {v:k for k, v in self.valid_data.dataset.vocab.items()}
        self.translator = Translator(
                self.model,
                
        )


    def eval(self, epoch):
        self.logger.info(f"Begin the #{epoch} epoch of model inference testing ~~~")
        self.model.eval()
        self.model.cpu()



    def decode_seq(self):
        pass







if __name__ == '__main__':
    pass