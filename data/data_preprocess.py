import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union, Optional
import os

import re

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder    



from collections import Counter
from datasets import (
    load_dataset,
    Dataset,
)

import sys
sys.path.append("../")

from config.config import DEVICE

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
        self.stopwords = stopwords.words('english')
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
        print("start prepare data ...")
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


    def __getitem__(self, idx)-> Dict[str, Union[torch.Tensor, str]]:
        """  
        获取单个样本  
        
        Args:  
            idx: 样本索引  
            
        Returns:  
            包含所有特征和目标变量的字典，格式为：  
            {  
                'numeric_feature_name': tensor(value, dtype=torch.float32),  
                'categorical_feature_name_encoded': tensor(value, dtype=torch.long),  
                'text_feature_name': str,  
                'target_name': tensor(value, dtype=torch.float32 or torch.long)  
            }  
        
        Raises:  
            IndexError: 当索引超出范围时  
            ValueError: 当数据格式不正确时  
        """  

        try:
            sample = {}

            # 获取数值型特征  
            if self.numeric_cols:  
                numeric_features:np.ndarray = self.df[self.numeric_cols].iloc[idx].values # values表示转为numpy
                # 确保numeric_features是二维数组  
                numeric_features = numeric_features.reshape(-1)  
                # 使用字典推导式提高代码简洁性  
                numeric_dict = {  
                    col: torch.tensor(val, dtype=torch.float32)
                    for col, val in zip(self.numeric_cols, numeric_features)  
                }  
                sample.update(numeric_dict) 
        
            
            # 获取类别型特征  
            if self.categorical_cols:  
                categorical_features:List[int]= [  
                    self.df[f"{col}_encoded"].iloc[idx]  # 是一个标量
                    for col in self.categorical_cols  
                ]

                # 使用字典推导式  
                categorical_dict = {  
                    f"{col}_encoded": torch.tensor(val, dtype=torch.long) 
                    for col, val in zip(self.categorical_cols, categorical_features)  
                }  
                sample.update(categorical_dict) 


            
            # 获取文本特征  
            if self.text_cols:  
                text_dict = {  
                    col: str(self.df[col].iloc[idx])  
                    for col in self.text_cols  
                }  

                sample.update(text_dict)
            
            # 获取目标变量  
            if self.target_col:  

                target_col = f"{self.target_col}_encoded"   \
                                    if self.target_col in self.label_encoders   \
                                                                    else self.target_col 
            
                target_value = self.df[target_col].iloc[idx]
 
                # 根据目标变量类型选择适当的张量类型  
                if isinstance(target_value, (int, np.integer)):  
                    target_tensor = torch.tensor(target_value, dtype=torch.long)  
                else:  
                    target_tensor = torch.tensor(target_value, dtype=torch.float32)  
                
                sample[self.target_col] = target_tensor  




            # 最后统一进行设备迁移，避免多次迁移  
            if hasattr(self, 'device') and self.device is not None:  
                sample = {  
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v  
                    for k, v in sample.items()  
                } 

            return sample

        except IndexError as e:
            raise IndexError(f"索引 {idx} 超出范围 (0, {len(self.df)-1})") from e

        except Exception as e:  
            raise ValueError(f"处理索引 {idx} 的数据时出错: {str(e)}") from e
    

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
    


    def add_avg_word(self):
        for i, col in enumerate(self.text_cols):
            self.df[f'avg_word_{col}'] = self.df[col].apply(lambda x: self.avg_word(x))

    

    def add_word_count(self, col:str):
        for i, col in enumerate(self.text_cols):
            self.df[f'word_count_{col}'] = self.df[col].apply(lambda x: len(x.split()))

    def add_char_count(self, col:str):
        for i, col in enumerate(self.text_cols):
            self.df[f'char_count_{col}'] = self.df[col].str.len()

    def to_lower(self):
        for i, col in enumerate(self.text_cols):
            self.df[col] = self.df[col].apply(lambda x: "".join([word.lower() for word in x.split()]))



    def to_upper(self):
        for i, col in enumerate(self.text_cols):
            self.df[col] = self.df[col].apply(
                lambda x: " ".join([word.upper() for word in x.split()])
            )


    def remove_punctuation_special_symbols(self):
        for i, col in enumerate(self.text_cols):
            self.df[col] = self.df[col].str.replace(r'[^\w\s]','')



    def remove_stopwords(self):
        for i, col in enumerate(self.text_cols):
            self.df[col] = self.df[col].apply(
                lambda x: " ".join([word for word in x.split() if word not in self.stopwords])
            )


    def remove_scarce_words(self):
        for i, col in enumerate(self.text_cols):
            freq = pd.Series(' '.join(self.df[col]).split()).value_counts()[-10:]

            self.df[col] = self.df[col].apply(
                lambda x: " ".join(x for x in x.split() if x not in freq)
            )


    def clean_data(self,   
                to_lower: bool = True,  
                remove_punct: bool = True,  
                remove_stops: bool = True,  
                remove_scarce: bool = True,  
                ) -> Dataset:  
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
            self.remove_scarce_words()  
            
        return self











class HFDataset(Dataset):  
    def __init__(
            self, 
            dataset_name: str, 
            text_column: str,
            subset:str = None, 
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

        # 复制原始数据集的属性  
        self._data =  self.dataset._data  
        self._info =  self.dataset.info  
        self._split =  self.dataset.split  
        self._features =  self.dataset.features  
        self._indices =  self.dataset._indices  

        # 调用父类的__init__  
        # super().__init__()  
        # super().__init__(  
        #     self.original_dataset.data,  
        #     self.original_dataset.info,  
        #     self.original_dataset.split,  
        #     self.original_dataset.features,  
        #     self.original_dataset._indices  
        # )  



    def __len__(self):
        return super().__len__()  


    def __getitem__(self, key: Union[int, slice, str])-> Union[Dict, List[Dict]]:
        """
        获取单个样本
        Args:
            key: 样本索引或切片
        Returns:
            处理后的样本
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            '''
            方法会根据数据集的长度调整切片的起始、结束和步长，
            返回一个包含调整后值的元组 (start, stop, step)

            总结来说，这句代码确保了切片操作在数据集的有效范围内，避免了越界错误。
            '''
            return [self._data[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            # 处理负数索引  
            if key < 0:  
                key = len(self) + key  
            if key < 0 or key >= len(self):  
                raise IndexError(f"Index {key} is out of range for dataset with size {len(self)}")  
            
            # return self._data[key].to_pydict() 
            return self._data[key]
        
        elif isinstance(key, str):
            if key not in self._features:  
                raise KeyError(f"Column '{key}' not found in dataset. Available columns: {list(self._features.keys())}")
            return self._data[key]
        
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
            lambda x: {self.text_column: x[self.text_column].lower()},   # x 的数据类型是一个字典（dict），它代表了数据集中的一行数据。
            remove_columns=[self.text_column]  
        )  
    
    def to_upper(self) -> None:  
        """将文本转换为大写"""  
        self.dataset = self.dataset.map(
            lambda x: {self.text_column: x[self.text_column].upper()},
            remove_columns = [self.text_column]
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







def test_hf_dataset():
    # 以IMDB数据集为例  
    hf_ds = HFDataset("./imdb", text_column="text", split = 'train')  
     # 打印原始数据集的一个样本  
    print("Original text:", hf_ds[hf_ds.text_column][:200])  

    cleaned_dataset = hf_ds.clean_data(  
        to_lower=True,  
        remove_punct=True,  
        remove_stops=True,  
        remove_scarce=True,  
        min_freq=5  
    )  
    
    # 打印清理后的样本  
    print("\nCleaned text:", cleaned_dataset[0][hf_ds.text_column][:200])  


    # 验证数据集的基本功能  
    print("\nDataset length:", len(cleaned_dataset))  
    print("Dataset features:", cleaned_dataset.features)




def test_pandas_dataset():
    text_cols = ["content", "title"]

    ds = PandasDataset('../sample_data.json', text_cols=text_cols)
    ds.print_dataframe_info()
    # print("Dataset Info = ", ds.info)
    ds.clean_data()
    print("clean dataset done")



if __name__ == '__main__':
    # ds = PandasDataset('../sample_data.json')
    # ds.print_dataframe_info()

    # test_hf_dataset()
    test_pandas_dataset()


    