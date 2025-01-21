import torch
import collections
import sys
import json


from transformer.Translator import Translator

from loader import load_data, DataGenerator

from torch.utils.data import DataLoader, Dataset

from collections import defaultdict



class Evaluator(object):    
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data:DataLoader = load_data(config['valid_data_path'],config,logger, shuffle=False)
        self.reverse_vocab = {v:k for k, v in self.valid_data.dataset.vocab.items()}
        self.translator = Translator(
                self.model,
                config['beam_size'],
                config['output_max_length'],
                config['pad_idx'],
                config['pad_idx'],
                config['start_idx'],
                config['end_idx']
        )


    def eval(self, epoch):
        self.logger.info(f"Begin the #{epoch} epoch of model inference testing ~~~")
        self.model.eval()
        self.model.cpu()

        self.stats_dict = defaultdict(int)  # store the evaluation results

        for index, batch_data in enumerate(self.valid_data):
            input_seqs, target_seqs, gold = batch_data
            for input_seq in input_seqs:
                input_seq: torch.LongTensor
                # 目前还无法同时翻译一批句子
                generate = self.translator.translate_sentence(input_seq.unsqueeze(0))    
                print("输入：",self.decode_seq(input_seq))
                print("输出：", self.decode_seq(input_seq))
                break   
        return
        


    def decode_seq(self, seq:torch.LongTensor):
        return "".join([self.reverse_vocab[int(idx)] for idx in seq])







if __name__ == '__main__':
    pass