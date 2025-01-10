import torch  
from torch import nn  
from torch.utils.data import Dataset, DataLoader  
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    T5Tokenizer, 
    T5ForConditionalGeneration  
)
from torch.optim import AdamW  
from typing import Dict, List, Optional  

class T5TrainingDataset(Dataset):  
    """  
    T5训练数据集类  
    """  
    def __init__(  
        self,   
        texts: List[str],      # 输入文本列表  
        targets: List[str],    # 目标文本列表  
        tokenizer: T5Tokenizer, # T5分词器  
        max_length: int = 512  # 最大序列长度  
    ):  
        self.texts = texts  
        self.targets = targets  
        self.tokenizer = tokenizer  
        self.max_length = max_length  

    def __len__(self):  
        return len(self.texts)  

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:  
        # 对输入文本进行编码  
        input_encoding = self.tokenizer(  
            self.texts[idx],  
            max_length=self.max_length,  
            padding='max_length',  
            truncation=True,  
            return_tensors="pt"  
        )  

        # 对目标文本进行编码  
        target_encoding = self.tokenizer(  
            self.targets[idx],  
            max_length=self.max_length,  
            padding='max_length',  
            truncation=True,  
            return_tensors="pt"  
        )  

        # 准备训练数据  
        return {  
            'input_ids': input_encoding['input_ids'].squeeze(),  
            'attention_mask': input_encoding['attention_mask'].squeeze(),  
            'labels': target_encoding['input_ids'].squeeze(),  
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()  
        }  

class T5Trainer:  
    """  
    T5模型训练器类  
    """  
    def __init__(  
        self,  
        model: T5ForConditionalGeneration,  
        tokenizer: T5Tokenizer,  
        device: str,  
        max_length: int = 512  
    ):  
        self.model = model.to(device)  
        self.tokenizer = tokenizer  
        self.device = device  
        self.max_length = max_length  

    def train_step(  
        self,  
        batch: Dict[str, torch.Tensor],  
        optimizer: torch.optim.Optimizer  
    ) -> float:  
        """  
        单步训练函数（包含Teacher Forcing）  
        """  
        # 将模型设置为训练模式  
        self.model.train()  
        
        # 将数据移到指定设备  
        input_ids = batch['input_ids'].to(self.device)  
        attention_mask = batch['attention_mask'].to(self.device)  
        labels = batch['labels'].to(self.device)  
        decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)  

        # 清零梯度  
        optimizer.zero_grad()  

        # 前向传播  
        # T5模型在训练时会自动使用Teacher Forcing  
        # labels参数会被用作decoder的输入（移位一位作为teacher forcing输入）  
        outputs = self.model(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            labels=labels,                     # 这里的labels会被自动处理用于teacher forcing  
            decoder_attention_mask=decoder_attention_mask  
        )  

        # 获取损失  
        loss = outputs.loss  

        # 反向传播  
        loss.backward()  

        # 梯度裁剪（防止梯度爆炸）  
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  

        # 更新参数  
        optimizer.step()  

        return loss.item()  

    def train(  
        self,  
        train_dataloader: DataLoader,  
        num_epochs: int,  
        optimizer: Optional[torch.optim.Optimizer] = None,  
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None  
    ):  
        """  
        完整训练循环  
        """  
        # 如果没有提供优化器，创建默认的AdamW优化器  
        if optimizer is None:  
            optimizer = AdamW(self.model.parameters(), lr=3e-4)  

        # 训练循环  
        for epoch in range(num_epochs):  
            total_loss = 0  
            for batch_idx, batch in enumerate(train_dataloader):  
                # 执行单步训练  
                loss = self.train_step(batch, optimizer)  
                total_loss += loss  

                # 打印训练进度  
                if batch_idx % 100 == 0:  
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx} - Loss: {loss:.4f}")  

                # 更新学习率  
                if scheduler is not None:  
                    scheduler.step()  

            # 计算epoch平均损失  
            avg_loss = total_loss / len(train_dataloader)  
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")  

def main():  
    """  
    主函数：展示如何使用训练器  
    """  
    # 初始化tokenizer和模型  
    tokenizer = T5Tokenizer.from_pretrained('t5-base')  
    model = T5ForConditionalGeneration.from_pretrained('t5-base')  

    # 示例数据  
    texts = [  
        "translate English to German: The house is wonderful",  
        "summarize: The article discusses various aspects of climate change"  
    ]  
    targets = [  
        "Das Haus ist wunderbar",  
        "The article is about climate change"  
    ]  

    # 创建数据集  
    dataset = T5TrainingDataset(  
        texts=texts,  
        targets=targets,  
        tokenizer=tokenizer  
    )  

    # 创建数据加载器  
    dataloader = DataLoader(  
        dataset,  
        batch_size=2,  
        shuffle=True  
    )  

    # 初始化训练器  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    trainer = T5Trainer(  
        model=model,  
        tokenizer=tokenizer,  
        device=device  
    )  

    # 创建优化器和学习率调度器  
    optimizer = AdamW(model.parameters(), lr=3e-4)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  

    # 开始训练  
    trainer.train(  
        train_dataloader=dataloader,  
        num_epochs=3,  
        optimizer=optimizer,  
        scheduler=scheduler  
    )  

if __name__ == "__main__":  
    main()
