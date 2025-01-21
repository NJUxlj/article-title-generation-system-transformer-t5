import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout= nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask = None):
        '''
        k.shape = (B, n_head, k_len, d_k)
        mask.shape = (B, n_head, q_len, k_len)
        '''

        attn = torch.matmul(q/self.temperature,k.transpose(2,3)) # shape = (B, n_head, q_len, k_len)


        if mask!=None:
            attn.masked_fill(mask==0,-1e9)
            # 如果传来一个0-1的上三角矩阵，attention矩阵中的整个上三角（不包括对角线）都要加上-1e9，softmax之后就变成了0
        

        attn = self.dropout(F.softmax(attn,dim=-1))

        output = torch.matmul(attn, v)

        return output, attn
