from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Set










'''
训练过程：

将文本切分成基础字符单位
统计字符对频率
迭代合并最频繁的字符对
更新词表和合并规则
分词过程：

将输入文本切分成基础单位
应用已学习的合并规则
输出最终的分词结果

'''





class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.vocab = set()
        self.merges = {}  # 存储合并规则
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def train(self, texts: List[str]):
        """训练BPE分词器"""
        # 1. 预处理文本，统计单词频率
        words = self._preprocess_texts(texts)
        
        # 2. 初始化词表为字符级别
        char_vocab = set()
        for word, freq in words.items():
            # 将每个单词切分成字符，用'</w>'标记词尾
            chars = ' '.join(list(word)) + ' </w>'
            self.word_freqs[chars] = freq
            char_vocab.update(chars.split())
        
        self.vocab = char_vocab
        
        # 3. 迭代合并最频繁的字符对
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            # 找出最频繁的字符对
            pairs = self._get_stats()
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # 更新词表和合并规则
            self.vocab.add(''.join(best_pair))
            self.merges[best_pair] = ''.join(best_pair)
            
            # 应用合并规则
            self._apply_merge(best_pair)
            
            if len(self.vocab) >= self.vocab_size:
                break
    
    def _preprocess_texts(self, texts: List[str]) -> Counter:
        """预处理文本并统计单词频率"""
        words = Counter()
        for text in texts:
            # 使用正则表达式进行基础分词
            tokens = [match.group() for match in self.pattern.finditer(text)]
            # 清理并统计词频
            for token in tokens:
                token = token.strip()
                if token:
                    words[token] += 1
        return words
    
    def _get_stats(self) -> Dict[Tuple[str, str], int]:
        """统计所有相邻字符对的频率"""
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def _apply_merge(self, pair: Tuple[str, str]):
        """应用一个合并规则到所有单词"""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        new_word_freqs = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            if bigram in word:
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            else:
                new_word_freqs[word] = freq
                
        self.word_freqs = new_word_freqs
    
    def tokenize(self, text: str) -> List[str]:
        """使用训练好的BPE模型对文本进行分词"""
        # 1. 预处理文本
        tokens = [match.group() for match in self.pattern.finditer(text)]
        result = []
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
                
            # 2. 将单词转换为字符序列
            word = ' '.join(list(token)) + ' </w>'
            
            # 3. 迭代应用合并规则
            while True:
                # 找出所有可能的字符对
                pairs = self._get_pairs(word)
                if not pairs:
                    break
                    
                # 找到第一个在合并规则中的字符对
                bigram = None
                for pair in pairs:
                    if pair in self.merges:
                        bigram = pair
                        break
                
                if not bigram:
                    break
                    
                # 应用合并规则
                p1, p2 = bigram
                word = word.replace(f"{p1} {p2}", ''.join(bigram))
            
            # 4. 移除</w>标记并添加到结果中
            word = word.replace('</w>', '')
            result.extend(word.split())
            
        return result
    
    def _get_pairs(self, word: str) -> Set[Tuple[str, str]]:
        """获取单词中的所有相邻字符对"""
        symbols = word.split()
        return {(symbols[i], symbols[i+1]) 
                for i in range(len(symbols)-1)}

# 使用示例
if __name__ == "__main__":
    # 示例文本
    texts = [
        "Hello world! This is a test.",
        "BPE is a data compression technique.",
        "It can be used for text tokenization.",
    ]
    
    # 初始化并训练分词器
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(texts)
    
    # 测试分词效果
    test_text = "Hello world!"
    tokens = tokenizer.tokenize(test_text)
    print(f"Original text: {test_text}")
    print(f"Tokenized: {tokens}")
