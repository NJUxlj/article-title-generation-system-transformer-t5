''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k*n_head, bias=False)
        self.w_ks = nn.Linear(d_model, d_k*n_head, bias=False)
        self.w_vs = nn.Linear(d_model, d_v*n_head, bias=False)
        self.fc = nn.Linear(d_v*n_head, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5)) # d_k ** 0.5

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


        

    def forward(self, q, k, v, mask=None):
        '''
        mask.shape = (B, q_len, k_len)
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_q, n_head, d_v)


        # Transpose for attention dot product: b x n x lq x dv
        q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        # 加掩码
        if mask is not None:
            # For head axis broadcasting.
            mask = mask.unsqueeze(1)      # mask.shape = (B, n_head, q_len, k_len)

        # 计算 attention
        q, attn = self.attention.forward(q, k ,v, mask=mask)
        q:torch.LongTensor
        # 转置回来
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q= q.transpose(1,2).contiguous().view(sz_b, len_q, -1)

        # dropout， residual， layernorm
        q = self.dropout(self.fc(q))
        q+=residual
        q = self.layer_norm(q)

        return q, attn



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        '''
        我们的编码器和解码器中的每一层都包含一个完全连接的前馈网络，
        该网络分别且相同地应用于每个位置。

        FFN(x) = max(0, xW1 + b1)W2 + b2
        其中：
            x(B, L, H1) * W1(H1, H2) = (B, L, H2)

        final_output = LayerNorm(x + FFN(x))
        '''
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in) # B L d_in
        self.layer_norm = nn.LayerNorm(normalized_shape=d_in)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x)->torch.LongTensor:
        residual = x

        x = F.relu(self.w_1(x))
        x = self.w_2(x)
        x =self.dropout(x)

        x = self.layer_norm(x + residual)

        return x
        


