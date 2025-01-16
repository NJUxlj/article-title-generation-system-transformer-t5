import torch
import torch.nn as nn

from transformer.SubLayers import MultiHeadAttention
from transformer.SubLayers import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
    


    def forward(self):
        pass







class DecoderLayer(nn.Module):
    def __init__(self, 
                 d_model, d_inner, n_head,d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention()


    def forward(self):
        pass









