
import numpy as np
from typing import List, Tuple, Dict
from queue import PriorityQueue
import torch
import torch.nn as nn

from transformer.Models import Transformer
from config.config import Config, BERT_PATH

from transformers import BertTokenizer

class BeamSearch:
    def __init__(self, model, beam_width: int = 3, max_length: int = 50, 
                 length_penalty: float = 0.6):
        """
        初始化 Beam Search
        
        Args:
            model: 用于生成序列的模型（例如Transformer）
            beam_width: 束宽，每一步保留的候选数量
            max_length: 生成序列的最大长度
            length_penalty: 长度惩罚因子
        """
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
        
    def _length_penalty_score(self, length: int) -> float:
        """
        计算长度惩罚
        score = (5 + length)^length_penalty / (5 + 1)^length_penalty
        """
        return ((5 + length) ** self.length_penalty) / (6 ** self.length_penalty)
    
    def _get_top_k_candidates(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取概率最高的k个候选
        
        Args:
            logits: 模型输出的logits，shape [vocab_size]
            k: 需要返回的候选数量
            
        Returns:
            Tuple[概率值, 词索引]
        """
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_ids = torch.topk(probs, k, dim=-1)
        return top_k_probs, top_k_ids
    
    def search(self, input_ids: torch.Tensor, 
               start_token_id: int, 
               end_token_id: int) -> List[Dict]:
        """
        执行beam search
        
        Args:
            input_ids: 输入序列
            start_token_id: 起始标记ID
            end_token_id: 结束标记ID
            
        Returns:
            List[Dict]: 包含生成序列及其得分的列表
        """
        # 初始化
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 初始序列只包含起始标记
        current_sequences = torch.full((batch_size, 1), start_token_id, 
                                    dtype=torch.long, device=device)
        
        # 用于存储完成的序列
        finished_sequences = []
        
        # 初始分数
        sequence_scores = torch.zeros(batch_size, device=device)
        
        for step in range(self.max_length):
            # 获取模型预测
            with torch.no_grad():
                outputs = self.model(current_sequences)
                logits = outputs.logits[:, -1, :]  # 获取最后一个时间步的输出
                
            vocab_size = logits.shape[-1]
            
            # 为每个序列获取top k个候选
            top_k_probs, top_k_ids = self._get_top_k_candidates(
                logits, min(self.beam_width, vocab_size)
            )
            
            # 扩展所有可能的候选
            all_candidates = []
            for idx in range(batch_size):
                for beam_idx in range(top_k_probs.shape[1]):
                    candidate_score = sequence_scores[idx] + \
                                    torch.log(top_k_probs[idx, beam_idx])
                    candidate_sequence = torch.cat([
                        current_sequences[idx],
                        top_k_ids[idx, beam_idx].unsqueeze(0)
                    ])
                    
                    # 检查是否生成了结束标记
                    if top_k_ids[idx, beam_idx] == end_token_id:
                        # 应用长度惩罚并保存完成的序列
                        final_score = candidate_score / self._length_penalty_score(
                            len(candidate_sequence)
                        )
                        finished_sequences.append({
                            'sequence': candidate_sequence.cpu().tolist(),
                            'score': final_score.item()
                        })
                    else:
                        all_candidates.append({
                            'sequence': candidate_sequence,
                            'score': candidate_score
                        })
            
            # 如果没有活跃的候选，或者所有序列都完成，则结束搜索
            if not all_candidates:
                break
                
            # 选择得分最高的beam_width个候选
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = all_candidates[:self.beam_width]
            
            # 更新当前序列和分数
            current_sequences = torch.stack([c['sequence'] for c in top_candidates])
            sequence_scores = torch.tensor([c['score'] for c in top_candidates], 
                                        device=device)
            
            batch_size = len(top_candidates)
            
        # 将未完成的序列也添加到结果中
        for idx in range(batch_size):
            final_score = sequence_scores[idx] / self._length_penalty_score(
                len(current_sequences[idx])
            )
            finished_sequences.append({
                'sequence': current_sequences[idx].cpu().tolist(),
                'score': final_score.item()
            })
        
        # 按分数排序并返回结果
        finished_sequences.sort(key=lambda x: x['score'], reverse=True)
        return finished_sequences

# 使用示例
def example_usage():
    """
    展示如何使用BeamSearch类的示例
    """
    # 假设我们有一个预训练的模型
    model = Transformer(

    )  
    
    # 初始化beam search
    beam_search = BeamSearch(
        model=model,
        beam_width=3,
        max_length=50,
        length_penalty=0.6
    )
    
    # 准备输入
    input_ids = torch.tensor([[1, 2, 3]])  # 示例输入序列
    start_token_id = 0
    end_token_id = 1
    
    # 执行beam search
    results = beam_search.search(
        input_ids=input_ids,
        start_token_id=start_token_id,
        end_token_id=end_token_id
    )
    
    # 打印结果
    for idx, result in enumerate(results):
        print(f"Sequence {idx + 1}:")
        print(f"Tokens: {result['sequence']}")
        print(f"Score: {result['score']:.4f}")
        print()

# 辅助函数：用于将token ID转换为文本
def decode_sequence(sequence, tokenizer=None):
    """
    将token ID序列转换为可读文本
    
    Args:
        tokenizer: 分词器
        sequence: token ID序列
        
    Returns:
        str: 解码后的文本
    """

    if tokenizer == None:
        tokenizer = BertTokenizer(BERT_PATH)
    return tokenizer.decode(sequence, skip_special_tokens=True)

