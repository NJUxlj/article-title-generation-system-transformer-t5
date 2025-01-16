import torch
import torch.nn as nn

from transformer.SubLayers import MultiHeadAttention
from transformer.SubLayers import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    '''compose with 2 sublayers'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    

    def forward(self, enc_input, slf_attn_mask=None):

        enc_output, enc_attn = self.slf_attn.forward(
            enc_input, enc_input, enc_input, mask = slf_attn_mask
        )

        enc_output = self.pos_ffn.forward(enc_output)

        return enc_output, enc_attn









class DecoderLayer(nn.Module):
    def __init__(self, 
                 d_model, d_inner, n_head,d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        # self.enc_attn 的作用
            # 将Decoder输入的Query中的每一行都用一个Attention矩阵去映射
            # 这个attention矩阵捕捉的是 query本身和encoder输出的完整上下文之间的关系
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None
            ):
        '''
        dec_input: decoder的输入
        enc_output: encoder最终输出的上下文
        slf_attn_mask: decoder最底下的 Masked multi-head attention 的掩码， 是上三角, 用来遮住未来的信息
        dec_enc_attn_mask: decoder中第二层的  multi-head attention 的掩码， 填 None就行
        '''
        dec_output, dec_slf_attn = self.slf_attn.forward(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )


        dec_output, dec_enc_attn = self.enc_attn.forward(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )


        dec_output = self.pos_ffn.forward(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn









