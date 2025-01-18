import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask




class Translator(nn.Module):
    def __init__(
            self,
            model:Transformer, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx,
            ):
        super().__init__()


        self.model = model
        self.model.eval()

        self.alpha=0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        '''
        register_buffer是一个方法，用于注册一个不需要梯度更新的缓冲区（buffer）。
        这些缓冲区通常用于存储模型的一些固定参数或状态，例如模型的初始序列、空白序列和长度映射等。
        '''
        # 这个缓冲区通常用于初始化生成序列的开始部分
        self.register_buffer('init_seq',torch.LongTensor([[self.trg_bos_idx]])) # shape = (1,1)

        # 这个调用注册了一个名为blank_seqs的缓冲区，它是一个形状为(beam_size, max_seq_len)的LongTensor，
        # 其中所有元素都被初始化为目标序列的填充标记（trg_pad_idx）。这个缓冲区通常用于存储生成序列的中间状态，
        # 其中每个序列都以开始标记开始，并在生成过程中逐步填充。
        self.register_buffer("blank_seqs", torch.full((self.beam_size, self.max_seq_len), fill_value=self.trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:,0] = self.trg_bos_idx
        
        # 这个调用注册了一个名为len_map的缓冲区，它是一个形状为(1, max_seq_len)的LongTensor，
        # 其中包含了从1到max_seq_len的连续整数。
        # 这个缓冲区通常用于 **计算生成序列的长度** ，以便在生成过程中进行 "长度惩罚" 等操作。
        self.register_buffer(
            "len_map",
            torch.arange(1, self.max_seq_len+1, dtype=torch.long).unsqueeze(0),
            ) # shape = (1, ,max_seq_len)

    

    def _model_decode(self, trg_seq, enc_output, src_mask):
        '''
        trg_seq: shape = (B,L), 这里 B==1
        enc_output: shape = (B, L, d_model)

        return logits, shape = (1,L,vocab_size)
        '''
        trg_mask = get_subsequent_mask(trg_seq) # shape = (1,L,L)
        decoder_output, *_=self.model.decoder.forward(
            trg_seq, trg_mask, enc_output, src_mask
        ) # shape = (1,L,d_model)

        decoder_logits = self.model.trg_word_prj(decoder_output) # shape = (1,L,vocab_size)

        logits = F.softmax(decoder_logits, dim=-1)
        return logits



    def _get_init_state(self, src_seq, src_mask):
        '''
        功能：
            _get_init_state 方法初始化了束搜索的状态，包括编码输出、生成序列的初始状态和得分。
            这些状态将用于后续的束搜索过程，以生成目标语言的序列。


            在束搜索（beam search）算法中，gen_seq 是用于存储生成序列的张量。
            将 gen_seq 的第二个位置设置为概率最高的词的索引，是因为第一个位置已经被初始化为目标语言的开始标记（例如 <s>）。
            这样做的目的是为了在束搜索的第一步中，选择概率最高的词作为生成序列的第一个词。

            具体来说，gen_seq 的形状通常是 (beam_size, max_seq_len)，
            其中 beam_size 是束搜索的宽度，max_seq_len 是生成序列的最大长度。
            在初始化时，gen_seq 的第一列（即第一个位置）被设置为开始标记的索引，
            而第二列（即第二个位置）则被设置为概率最高的词的索引。

            在束搜索的后续步骤中，会根据当前的生成序列和编码器的输出，
            计算每个可能的下一个词的概率，并选择概率最高的 beam_size 个词作为下一步的候选词。
            这些候选词将被添加到 gen_seq 的下一列中，从而逐步生成完整的目标序列。
            总结来说，将 gen_seq 的第二个位置设置为概率最高的词的索引，是束搜索算法中的一个关键步骤，
            它为生成序列的生成提供了一个初始的起点。
        '''
        
        beam_size = self.beam_size

        enc_output, *_=self.model.encoder.forward(src_seq, src_mask)
        
        # self.init_seq.shape = (1,1)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask) # shape = (1, L, vocab_size)
        # dec_output[:, -1, :].shape = (batch_size, vocab_size)  可以看做从 vocab_size 个概率中挑出 topk个概率
        # best_k_probs.shape = (batch_size, beam_size)
        # best_k_idx.shape = (batch_size, beam_size), where batch_size = 1
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        '''
        从解码输出的最后一个时间步（即最后一个词）中，
            选择概率最高的 beam_size 个词及其对应的概率
        '''
        if scores.size(0)==1:
            scores = torch.log(best_k_probs).view(beam_size)
        else:
            # 这里还没做适配
            scores = torch.log(best_k_probs) # 对数概率在束搜索中用于计算得分。

        
        # self.blank_seqs 通常是一个填充了目标语言的填充标记（例如 <pad>）的张量，用于存储生成的序列。
        gen_seq = self.blank_seqs.clone().detach() # shape = (beam_size, max_seq_len)

        # 将 gen_seq 的第二个位置（即第一个生成的词）设置为概率最高的词的索引。
        gen_seq[:,1] = best_k_idx[0]


        # 将编码输出 enc_output 在第一个维度上重复 beam_size 次，以匹配束搜索的宽度。
        enc_output = enc_output.repeat(beam_size, 1, 1)

        return enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step ):
        ''''
        :param: gen_seq: 目前为止已经生成的序列, gen_seq.shape = (batch_size, seq_len)
        :param: scores: 目前为止已经生成的序列的分数, scores.shape = (batch_size, seq_len), 其中scores[i, j]表示第i个句子的第j个词的分数, 分数是log概率
        :param: dec_output: 模型的输出, dec_output.shape = (batch_size, seq_len, vocab_size)
        '''
        assert len(scores.size())==1, "we can pnly accept batch_size equals to 1 in this function"

        beam_size = self.beam_size



    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0)==1
        


