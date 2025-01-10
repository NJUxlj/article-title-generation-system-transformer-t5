import torch
import torch.nn as nn
import numpy as np



from transformer.Layers import EncoderLayer, DecoderLayer

class PositionalEncoding(nn.Module):
    

    def __init__(self, d_hid, n_position=200):
        pass


    def _get_sinusoid_encoding_table():
        

        def get_position_angle_vec(position):
            pass



    def forward(self, x):
        pass





class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        super().__init__()
        self.src_word_embed = nn.Embedding()
        self.position_enc = PositionalEncoding()
        self.layer_stack = nn.ModuleList([
            EncoderLayer()
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(p = dropout)
        self.layer_norm =  nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model






class Decoder(nn.Module):
    def __init__(
        self,):
        super().__init__()




class Transformer(nn.Module):

    def __init__(self):
        super().__init__()




    def forward(self):
        pass