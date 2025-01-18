import torch
import torch.nn as nn

import json
from config.config import Config
from torch.utils.data import Dataset, OrderedDict, DataLoader
from collections import defaultdict


from typing import List, Tuple, Dict


class DataGenerator(Dataset):
    def __init__(self, data_path, config, Logger):
        '''
        内部维护的data的数据类型：List[List[torch.LongTensor]]
        
        '''
        self.config = config
        self.logger = Logger
        self.path = data_path
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.config['pad_id'] = '[PAD]'
        self.config['start_id'] = '[CLS]'
        self.config['end_id'] = '[SEP]'
        self.load()


    def load(self):
        self.data:List[List[torch.LongTensor]] = []

        with open(file=self.path, mode='r',encoding = 'utf-8') as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line['title']
                content = line['content']
                self.prepare_data(title, content)
        return
        



    def encode_sentence(self, text, max_length, with_cls_token = True, with_seq_token = True)->List[int]:
        input_id = []

        if with_cls_token:
            input_id+=[self.vocab.get('[CLS]')]

        for char in text:
            input_id += [self.vocab.get(char,self.vocab['[UNK]'])]

        if with_seq_token:
            input_id += [self.vocab.get('[SEP]')]

        input_id = self.padding(input_id, max_length)
        return input_id


    def padding(self, input_id:List, length)->List:
        
        # truncate
        input_id  = input_id[:length]

        # padding
        input_id += [self.vocab.get('[PAD]')]*(length - len(input_id))
        return input_id



    def prepare_data(self, title, content)->List[List[torch.LongTensor]]:
        input_seq:List[int] = self.encode_sentence(content, self.config['input_max_length'], False, False)
        output_seq  =self.encode_sentence(title, self.config['output_max_length'])
        gold = self.encode_sentence(title, self.config['output_max_length'])

        self.data.append(
            [
                torch.LongTensor(input_seq), 
                torch.LongTensor(output_seq), 
                torch.LongTensor(gold)
            ]
        )


    def __getitem__(self, idx)->List[torch.LongTensor]:
        return self.data[idx] 
    


    def __len__(self):
        return len(self.data)





def load_vocab(vocab_path):
    token_dict = {}

    with open(file=vocab_path, mode='r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            token_dict[token] = idx
    return token_dict





#用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path,  config, logger)
    dataloader = DataLoader(dataset = dg, batch_size = config['batch_size'],  shuffle = shuffle)
    return dataloader