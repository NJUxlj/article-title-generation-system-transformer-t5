import torch
import torch.nn as nn
import numpy as np



from transformer.Layers import EncoderLayer, DecoderLayer


def get_pad_mask(seq, pad_idx):
    '''
    :param seq.shape = (B, L)

    :return shape = (B, L, )

    seq != pad_idx：这是一个逐元素的比较操作，
    比较序列 seq 中的每个元素是否不等于填充索引 pad_idx。
    结果是一个布尔型的张量，形状与 seq 相同

    .unsqueeze(-2)：这个操作在张量的倒数第二个维度上增加一个维度。
        这是为了将来与下三角掩码矩阵 (subsequent mask, shape = (B, L, L)) 做逐元素的 “与” 操作
    
    例如，如果 seq 的形状是 (batch_size, seq_length)，
    那么 (seq != pad_idx) 的形状也是 (batch_size, seq_length)，
    而 .unsqueeze(-2) 操作后，形状变为 (batch_size, 1, seq_length)。

    举例：
    True, True, True, False, False,
    True, True, True, False, False,    
    True, True, True, True, False,
    True, True, True, False, False, 
    '''
    return (seq!=pad_idx).unsqueeze(-2)



def get_subsequent_mask(seq:torch.LongTensor) -> torch.BoolTensor:
    '''
    生成一个掩码（mask），用于在序列模型（如Transformer）中屏蔽掉未来的信息。
    
        在序列到序列（seq2seq）模型中，
        解码器在生成输出时不应该看到未来的信息，
        因此需要一个掩码来确保解码器只能关注当前位置及其之前的位置。
    
    :param seq seq.shape = (B, L)


    subsequent_mask = (
        1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
        ).bool()

        这是生成掩码的核心代码。
        torch.triu 函数生成一个上三角矩阵，
        其中对角线及以上的元素为1，对角线以下的元素为0。
        
        diagonal=1 参数表示从主对角线向上偏移1，即主对角线上的元素也为0。
        1 - torch.triu(...) 将上三角矩阵中的1变为0，0变为1，
        得到一个下三角矩阵，其中对角线及以下的元素为1，对角线以上的元素为0。
        .bool() 将矩阵中的元素转换为布尔类型，即1变为 True，0变为 False。
        最终得到的 subsequent_mask 是一个形状为 (1, len_s, len_s) 的布尔类型张量，表示一个掩码，用于屏蔽掉未来的信息。

    
        举例：
        True, False, False, False, False,
        True, True, False, False, False,   
        True, True, True, False, False,
        True, True, True, True, False,
        True, True, True, True, True,

    '''
    len_s = seq.size(1)
    subsequent_mask = (
        1-torch.triu(torch.ones((1, len_s,len_s), device = seq.device), diagonal=1)
        ).bool()

    return subsequent_mask




class PositionalEncoding(nn.Module):
    

    def __init__(self, d_hid, n_position=200):
        '''
        '''
        # Add a buffer to the module.
        # buffer 中存放的是当前模块中的一些状态值，而非参数
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        '''

        register_buffer 方法的主要作用有以下几点：

            存储模型状态：缓冲区通常用于存储模型的状态信息，这些信息在模型的训练和推理过程中需要保持不变，
                        但又不是模型的可学习参数。例如，在循环神经网络（RNN）中，隐藏状态（hidden state）通常被存储在缓冲区中。

            避免梯度计算：由于缓冲区不是模型的参数，它们不会参与梯度计算。
                        这意味着在反向传播过程中，缓冲区不会被更新，从而节省了计算资源。

            模型保存和加载：缓冲区会随着模型一起被保存和加载。这使得模型在重新加载时能够恢复到之前的状态，包括缓冲区中的数据。

        这行代码注册了一个名为 pos_table 的缓冲区，它存储了正弦位置编码表。
                    这个表在模型的前向传播过程中会被使用，
                    用于为输入序列中的每个位置添加位置信息，但它本身不会被优化器更新。
        '''

    def _get_sinusoid_encoding_table(self, n_positions, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position/(10000**(2*(i//2)/d_hid)) for i in range(d_hid)]

        # sinusoid_table.shape = (n_positions, d_hid)
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_positions)])

        # Apply sin to even indices along the d_hid dimension for each position vector; 2i
        sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])   # dim 2i
        sinusoid_table[:,1::2] = np.sin(sinusoid_table[:, 1::2])  # dim 2i+1

        # clone().detach(): 创建一个独立于原始张量的新张量，并在计算图中断开连接（禁用梯度计算）
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


    def forward(self, x):
        '''
        x.shape = (B, L, d_model)
        '''
        return x + self.pos_table(x[:,:x.size(1),:]).clone().detach()





class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(p = dropout)
        self.layer_norm =  nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        '''
        :param src_seq: padded input sequence of shape (B, L, d_model)
        :param src_mask: source sequence mask of shape (B, L)
        '''
        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)

        if self.scale_emb:
            enc_output*= self.d_model*0.5
        

        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_layer: EncoderLayer
            enc_output, enc_slf_attn = enc_layer.forward(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,



class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
        self, n_trg_vocab,
        d_word_vec, n_layers,n_head, d_k, d_v,
        d_model, d_inner, pad_idx, n_position=200,dropout=0.1, scale_emb=False
        ):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position)
        self.dropout =nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList(
            [
              DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)  for _ in range(n_layers)
            ]
        )
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)



    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        
        dec_slf_attn_list, dec_enc_attn_list = [], []

        decoder_output = self.trg_word_emb(trg_seq)

        if self.scale_emb:
            decoder_output *= self.d_model*0.5
        decoder_output = self.dropout(self.position_enc(decoder_output))
        decoder_output = self.layer_norm(decoder_output)


        for dec_layer in self.layer_stack:
            dec_layer: DecoderLayer
            dec_output,dec_slf_attn, dec_enc_attn = dec_layer.forward(
                decoder_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []


        if return_attns:
            return decoder_output, dec_slf_attn_list, dec_enc_attn_list
        return decoder_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self,
            n_src_vocab,n_trg_vocab,
            src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512,d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_positions =200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='none',
            ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['none', 'emb','prj']
        # scale embedding的前提是， embedding 与 linear projection 共享权重
        scale_emb:bool = (scale_emb_or_prj=="emb") if trg_emb_prj_weight_sharing else False
        self.scale_prj:bool = (scale_emb_or_prj=="prj") if trg_emb_prj_weight_sharing else False
        self.d_model = d_model


        self.encoder = Encoder(
            n_src_vocab = n_src_vocab, n_postions = n_positions,
            d_word_vec=d_word_vec, d_model = d_model, d_inner=d_inner,
            n_layers =n_layers, n_head =n_head, d_k=d_k, d_v = d_v,
            pad_idx = src_pad_idx, dropout=dropout,scale_emb = scale_emb
        )

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_positions=n_positions,
            d_word_vec=d_word_vec, d_model = d_model, d_inner=d_inner,
            n_layers =n_layers, n_head =n_head, d_k=d_k, d_v = d_v,
            pad_idx = trg_pad_idx, dropout=dropout,scale_emb = scale_emb
        )       

        # 也是一个word embedding
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

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

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
        the dimensions of all module outputs shall be the same.'
        

        if trg_emb_prj_weight_sharing:
            # share the weight between the target word embedding and the last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight
            
    def forward(self, src_seq, trg_seq):
        src_mask  = get_pad_mask(src_seq, self.src_pad_idx) # shape = [B, 1, L]
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq) # shape = [1, L, L]

        '''
        get_pad_mask(trg_seq, self.trg_pad_idx) 生成一个布尔型的掩码，
        用于标记目标序列中的填充位置，与源序列的掩码生成方式相同

        shape = [B, 1, L] 的 bool 矩阵, 最后会广播成 [B, L, L]

        1 1 1 1 pad pad pad  

        ----> 广播以后

        1 1 1 1 pad pad pad
        1 1 1 1 pad pad pad
        1 1 1 1 pad pad pad
        1 1 1 1 pad pad pad
        1 1 1 1 pad pad pad
        1 1 1 1 pad pad pad
        1 1 1 1 pad pad pad


        get_subsequent_mask(trg_seq) 生成一个下三角矩阵的掩码，
        用于防止解码器在训练时看到未来的信息。
        这个掩码的形状为 (batch_size, seq_len, seq_len)，
        其中对角线及以下的元素为 True，对角线以上的元素为 False。

        shape = [1, L, L] 的 bool矩阵

        1 0 0 0 0 
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1

        & 操作符用于对两个掩码进行逐元素的逻辑与操作，得到最终的 trg_mask。
        这个掩码的形状为 (batch_size, seq_len, seq_len)，
        其中 True 表示该位置是有效输入（不是 padding id）且在当前时间步之前，
        False 表示该位置是填充或在当前时间步之后。

        L 是decoder的目标字符序列的长度
        '''


        # enc_output, *_ = ... 这部分代码使用了Python的解包（unpacking）语法。
        # enc_output会接收self.encoder返回的第一个值，而*_会接收剩余的所有值（如果有的话）并丢弃。
        enc_output, *_ = self.encoder.forward(
            src_seq = src_seq,
            src_mask = src_mask
        )

        dec_output, *_ = self.decoder.forward(
            trg_seq, trg_mask, enc_output, src_mask
        )

        seq_logit = self.trg_word_prj(dec_output)

        if self.scale_prj:
            seq_logit *= self.d_model**-0.5
        
        seq_logit = seq_logit.view(-1, seq_logit.shape[-1])
        return seq_logit