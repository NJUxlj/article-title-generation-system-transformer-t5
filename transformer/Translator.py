import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask

from typing import List, Dict, Tuple



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
        return 
            enc_output, gen_seq, scores
            
            enc_output.shape = (batch_size * beam_size, L, d_model), where batch_size = 1
            gen_seq.shape = (beam_size, L)
        功能：
            _get_init_state 方法初始化了束搜索的状态，包括编码输出、生成序列的初始状态和得分。
            这些状态将用于后续的束搜索过程，以生成目标语言的序列。


            在束搜索（beam search）算法中，gen_seq 是用于存储生成序列的张量。
            将 gen_seq 的第二个位置设置为概率最高的词的索引，是因为第一个位置已经被初始化为目标语言的开始标记（例如 <s>）。
            这样做的目的是为了在束搜索的第一步中，选择概率最高的词作为生成序列的第一个词。

            因此，gen_seq在初始化时，提前设置了两个位置的值；token_id = 0, token_id = 1

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
        # best_k_idx.shape = (batch_size, beam_size), where batch_size = 1, so it is in fact (1, beam_size)
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        '''
        从解码输出的最后一个时间步（即最后一个词）中，
            选择概率最高的 beam_size 个词及其对应的概率
        '''
        if scores.size(0)==1:
            # 先转为对数概率，再展平为 shape = (beam_size, )
            scores = torch.log(best_k_probs).view(beam_size)
        else:
            # 这里还没做适配
            scores = torch.log(best_k_probs) # 对数概率在束搜索中用于计算得分。

        
        # self.blank_seqs 通常是一个填充了目标语言的填充标记（例如 <pad>）的张量，用于存储生成的序列。
        gen_seq = self.blank_seqs.clone().detach() # shape = (beam_size, max_seq_len)

        # 将 gen_seq 的第二个位置（即第一个生成的词）设置为概率最高的词的索引。
        gen_seq[:,1] = best_k_idx[0]


        # 将编码输出 enc_output 在第一个维度上重复 beam_size 次，以匹配束搜索的宽度。
        # repeat 方法接受三个参数，分别表示在每个维度上的重复次数。
        enc_output = enc_output.repeat(beam_size, 1, 1) # shape = (beam_size, L, d_model)

        '''
        这句代码的作用是将编码器的输出 enc_output 在第一个维度上重复 beam_size 次，
        而保持其他维度不变。

        具体来说，repeat 方法接受三个参数，分别表示在每个维度上的重复次数。
        在这里，beam_size 是束搜索的宽度，因此 enc_output 在第一个维度上被重复了 beam_size 次，
        以匹配束搜索的宽度。

        在束搜索算法中，通常会生成多个候选序列，每个候选序列都有一个对应的得分。
        为了能够同时处理多个候选序列，编码器的输出需要在第一个维度上进行扩展，
        以匹配候选序列的数量。这样，每个候选序列都可以使用相同的编码器输出进行解码，
        从而生成相应的目标序列。

        总结来说，enc_output = enc_output.repeat(beam_size, 1, 1) 
        这句代码的作用是将编码器的输出扩展到束搜索的宽度，由于我们的这里的batch_size指定是1， 
        所以 enc_output.shape 实际= (1 * beam_size, L, d_model)
        以便在束搜索过程中同时处理多个候选序列。

        例如，原始的enc_output.shape = (1, L, d_model), 经过repeat之后, enc_output.shape = (beam_size, L, d_model)

        或者，原始的enc_output.shape = (B, L, d_model), 经过repeat之后, enc_output.shape = (B * beam_size, L, d_model)
        '''

        return enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output:torch.LongTensor, scores:torch.LongTensor, step ):
        '''
        这个函数实现了beam search的核心逻辑：
            维护多个可能的序列（beam），
            在每一步都扩展这些序列，然后选择最有可能的k个序列继续进行下一步的生成。
            这样可以比贪心搜索得到更好的生成结果。

        :param: gen_seq: 目前为止已经生成的序列, gen_seq.shape = (batch_size, seq_len)
        :param: scores: 目前为止已经生成的序列的分数, scores.shape = (batch_size, seq_len), 其中scores[i, j]表示第i个句子的第j个词的分数, 分数是log概率
        :param: dec_output: 模型的输出, dec_output.shape = (beam_size, seq_len, vocab_size), 
        '''
        assert len(scores.size())==1, "we can pnly accept batch_size equals to 1 in this function"

        beam_size = self.beam_size
        # Get k candidates for each beam (i.e., each token in the max_length), k^2 candidates in total.
        # dec_output[:,-1,:].shape = (beam_size, vocab_size)

        # dec_output[:, -1, :] 取最后一个时间步的输出
        # topk(beam_size) 为每个beam选择概率最高的k个候选词
        # 结果形状: best_k2_probs, best_k2_idx 都是 [beam_size, beam_size]
        best_k2_probs, best_k2_idx = dec_output[:,-1,:].topk(beam_size) # shape = (beam_size, beam_size)


        # Include the previous scores
        '''
        将概率转换为对数概率
        scores.view(beam_size, 1) 将之前的分数改变形状以便广播
        将新的对数概率加到之前的累积分数上
        结果形状: [beam_size, beam_size]
        '''
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        # scores.view(-1) 将分数展平为一维，大小为 beam_size * beam_size
        # 从k^2个候选中选择最高的k个分数和它们的索引。    
        # best_k_idx_in_k2 包含这k个最佳候选的索引
        scores, best_k_idx_in_k2= scores.view(-1).topk(beam_size) # shape = (beam_size, )


        # Get the corresponding positions of the best k candidiates.
        # 将当前取出的k个最佳id，映射到词表空间的真实下标。

        # best_k_r_idxs: 表示在哪个beam中
        # best_k_c_idxs: 表示在该beam的哪个候选位置
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size

        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        '''
        更新生成序列
        首先复制被选中的beam的历史序列
        然后在当前步骤位置填入新选择的词的索引
        '''
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]  
        gen_seq[:, step] = best_k_idx  




    def translate_sentence(self, src_seq)->List[int]:
        '''
        作用：
            将输入的源语言句子翻译成目标语言句子。
            它使用了束搜索（beam search）算法来生成目标语言句子，以提高翻译的质量。

        :param src_seq, shape = (batch_size, seq_len)

        :return LongTensor, shape = (seq_len, ) 返回一个目标语言句子的token id列表
        '''
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0)==1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 
        
        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx) # shape = (B, L)
            enc_output, gen_seq, scores=self._get_init_state(src_seq, src_mask)

            # gen_seq.shape = (beam_size, seq_len), 初始时， 

            ans_idx = 0 # default

            for step in range(1, max_seq_len):
                dec_output = self._model_decode(gen_seq[:,:step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx # shape = (beam_size, seq_len)

                # -- replace the eos with its position for the length penalty use
                # self.len_map.shape = (1, max_seq_len)
                # 包含了位置索引，比如 [0, 1, 2, 3, ..., max_seq_len-1]
                    # 用于追踪序列中每个位置的实际位置编号
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)

                # seq_lens.shape = (beam_size, )

                '''
                masked_fill(mask, value)
                    对于 mask 中为 True 的位置，用 value 值替换原来的值
                    在这里，所有不是EOS的位置(~eos_locs为True的位置)都被填充为 max_seq_len
                    而EOS位置保持原来的位置索引不变

                .min(1) 操作：
                    在维度1（序列长度维度）上取最小值
                    返回两个值：最小值和对应的索引

                整句代码的作用：找出所有生成序列中的结束标记位置
                '''

                # -- check if all beams contain eos
                # sum(1) 在维度1（序列长度维度）上求和
                # 对每个序列（beam）计算其中EOS标记的数量
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    
                    # length punishment
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
            
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()