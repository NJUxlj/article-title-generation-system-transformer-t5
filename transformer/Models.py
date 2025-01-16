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
        self.src_word_embed = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
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
    ''' A decoder model with self attention mechanism. '''

    def __init__(
        self, n_trg_vocab,
        d_word_vec, n_layers,n_head, d_k, d_v,
        d_model, d_inner, pad_idx, n_position=200,dropout=0.1, scale_emb=False
        ):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec)
        self.dropout =nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList(
            [
              DecoderLayer()  for _ in range(n_layers)
            ]
        )



class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self,
            n_src_vocab,
            d_word_vec=512, d_model=512,d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_positions =200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='none',
            ):
        super().__init__()



        assert scale_emb_or_prj in ['none', 'emb','prj']
        scale_emb = 
        self.scale_prj = () if else 
        self.d_model = d_model


        self.encoder = Encoder(

        )

        self.decoder = Decoder(

        )

        self.trg_word_prj = nn.Linear()

        # 模型参数初始化：
        '''
        这段代码的作用是对模型中的所有参数进行初始化。
        具体来说，它遍历了模型的所有参数，并对那些维度大于1的参数（通常是权重矩阵）应用了Xavier均匀初始化。

        Xavier初始化是一种常用的初始化方法，旨在保持各层输出的方差一致，避免梯度消失或爆炸问题。
        这种初始化方法根据输入和输出的维度来确定初始化的范围，使得初始化后的权重既不会太小导致梯度消失，也不会太大导致梯度爆炸。

        在这段代码中，p.dim() > 1 的条件确保了只有权重矩阵会被初始化，
        而偏置项（通常是一维的）则不会被初始化。
        '''

        '''
        在PyTorch中，nn.Module类有一个名为parameters()的方法，它返回一个生成器，
            生成该模块及其所有子模块中的所有可学习参数（即权重和偏置）。

        这是因为nn.Module类在内部维护了一个名为_parameters的字典，
            其中包含了该模块的所有可学习参数。
        '''

        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_seq, trg_seq):
        pass