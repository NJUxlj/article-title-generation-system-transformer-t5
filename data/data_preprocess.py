import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union, Optional
import os

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder    



from collections import Counter
from datasets import (
    load_dataset
)

from config import DEVICE

class PandasDataset(Dataset):
    '''
    json line 数据集包装类
    '''
    def __init__(
            self, 
            json_path:str,
            numeric_cols: List[str] = None,  
            categorical_cols: List[str] = None,  
            text_cols: List[str] = None,  
            target_col: str = None,  
            scaler_type: str = 'standard'
        ):
        self.df = self.load_json(json_path)
        self.name = os.path.basename(json_path)
        self.device = DEVICE

        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.text_cols = text_cols or []
        self.target_col = target_col
        self.scaler_type = scaler_type.lower() # 'standard' 或 'minmax'
        self.scaler = StandardScaler() if scaler_type=='standard' else MinMaxScaler()

        # 初始化特征处理器  
        self.label_encoders = {}
        '''
        StandardScaler:
            将数据转换为均值为0，标准差为1的分布
            适用于数据大致呈正态分布的情况
            对异常值敏感度较低

        MinMaxScaler:
            将数据缩放到[0,1]区间
            适用于数据分布不是正态分布的情况
            保留零值
            对异常值敏感
        '''

        self._prepare_data()
    

    def load_json(self, json_path:str)->pd.DataFrame:
        df = pd.read_json(json_path, lines=True)
        return df
    

    def _prepare_data(self):
        """预处理数据：标准化数值特征，编码类别特征"""
        # 处理数值型特征  
        if self.numeric_cols:  
            # 将数值特征转换为float32类型
            self.df[self.numeric_cols] = self.df[self.numeric_cols].astype(np.float32)
            # 简单的标准化处理（也可以使用StandardScaler或MinMaxScaler）  
            # self.numeric_mean = self.df[self.numeric_cols].mean()  
            # self.numeric_std = self.df[self.numeric_cols].std()  
            # self.df[self.numeric_cols] = (self.df[self.numeric_cols] - self.numeric_mean) / self.numeric_std

            self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])

        if self.categorical_cols:
            for col in self.categorical_cols:
                le = LabelEncoder() # LabelEncoder 是一个用于将分类变量编码为数值变量的工具
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le


        if self.target_col:
            if self.df[self.target_col].dtype == 'object':
                le = LabelEncoder()
                self.df[f'{self.target_col}_encoded'] = le.fit_transform(self.df[self.target_col]) # 将类别标签转为数值编码
                self.label_encoders[self.target_col] = le

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index)-> Dict[str, torch.Tensor]:
        """  
        获取单个样本  
        返回一个字典，包含所有特征和目标变量（如果存在）  
        """  
        sample = {}

        if self.numeric_cols:
            pass


        if self.categorical_cols:
            pass


        if self.text_cols:
            pass


        if self.target_col:
            pass



        return sample
    

    def print_data_features(self):
        pass


    def print_dataframe_info(self):
        print(f"\n{self.name} 信息:")  
        print("Shape:", self.df.shape)  
        print("\n前几行数据:")  
        print(self.df.head())  
        print("\n数据类型:")  
        print(self.df.dtypes)


    def avg_word(self, sentence:str):
        words = sentence.split()
        return (sum(len(word) for word in words)/len(words))


    def to_lower(self):
        pass



    def to_upper(self):
        pass


    def remove_punctuation_special_symbols(self):
        pass


    def remove_stopwords(self):
        pass


    def remove_scarce_words(self):
        pass



    def clean_data(self):
        pass











class HFDataset(Dataset):  
    def __init__(
            self, 
            dataset_name: str, 
            text_column: str, 
            split: str = "train",
            **kwargs
            ):  
        """  
        初始化HFDataset类  
        Args:  
            dataset_name: Hugging Face数据集名称  
            text_column: 需要处理的文本列名  
            split: 数据集分片名称  
        """  
        self.dataset = load_dataset(dataset_name, split=split)  
        self.text_column = text_column  
        self.stop_words = set(stopwords.words('english'))  


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        return self.dataset[index]
        
    def avg_word(self, sentence: str) -> float:  
        """  
        计算句子中单词的平均长度  
        Args:  
            sentence: 输入句子  
        Returns:  
            平均单词长度  
        """  
        words = sentence.split()  
        if not words:  
            return 0  
        return sum(len(word) for word in words) / len(words)  
    
    def to_lower(self) -> None:  
        """将文本转换为小写"""  
        self.dataset = self.dataset.map(  
            lambda x: {self.text_column: x[self.text_column].lower()}  
        )  
    
    def to_upper(self) -> None:  
        """将文本转换为大写"""  
        self.dataset = self.dataset.map(  
            lambda x: {self.text_column: x[self.text_column].upper()}  
        )  
    
    def remove_punctuation_special_symbols(self) -> None:  
        """移除标点符号和特殊字符"""  
        def clean_text(text: str) -> str:  
            # 保留字母、数字和空格，移除其他字符  
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)  
        
        self.dataset = self.dataset.map(  
            lambda x: {self.text_column: clean_text(x[self.text_column])}  
        )  
    
    def remove_stopwords(self) -> None:  
        """移除停用词"""  
        def remove_stops(text: str) -> str:  
            word_tokens = word_tokenize(text)  
            filtered_text = [word for word in word_tokens if word.lower() not in self.stop_words]  
            return ' '.join(filtered_text)  
        
        self.dataset = self.dataset.map(  
            lambda x: {self.text_column: remove_stops(x[self.text_column])}  
        )  
    
    def remove_scarce_words(self, min_freq: int = 5) -> None:  
        """  
        移除低频词  
        Args:  
            min_freq: 最小词频阈值  
        """  
        # 统计所有单词的频率  
        word_freq = Counter()  
        for example in self.dataset:  
            words = word_tokenize(example[self.text_column])  
            word_freq.update(words)  
        
        # 获取频率达到阈值的词集合  
        valid_words = {word for word, freq in word_freq.items() if freq >= min_freq}  
        
        def filter_scarce_words(text: str) -> str:  
            words = word_tokenize(text)  
            filtered_words = [word for word in words if word in valid_words]  
            return ' '.join(filtered_words)  
        
        self.dataset = self.dataset.map(  
            lambda x: {self.text_column: filter_scarce_words(x[self.text_column])}  
        )  
    
    def clean_data(self,   
                   to_lower: bool = True,  
                   remove_punct: bool = True,  
                   remove_stops: bool = True,  
                   remove_scarce: bool = True,  
                   min_freq: int = 5) -> Dataset:  
        """  
        执行完整的数据清理流程  
        Args:  
            to_lower: 是否转换为小写  
            remove_punct: 是否移除标点符号  
            remove_stops: 是否移除停用词  
            remove_scarce: 是否移除低频词  
            min_freq: 最小词频阈值  
        Returns:  
            清理后的数据集  
        """  
        if to_lower:  
            self.to_lower()  
        if remove_punct:  
            self.remove_punctuation_special_symbols()  
        if remove_stops:  
            self.remove_stopwords()  
        if remove_scarce:  
            self.remove_scarce_words(min_freq)  
            
        return self.dataset  
    
    def get_dataset(self) -> Dataset:  
        """获取当前数据集"""  
        return self.dataset  










if __name__ == '__main__':
    ds = PandasDataset('../sample_data.json')
    ds.print_dataframe_info()