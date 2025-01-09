## T5 
T5是由Google Research在2019年提出的一个里程碑式的模型 []()。它的核心理念是将所有NLP任务统一转换为"文本到文本"的格式，这是一个非常优雅的设计思想。

---
## 2. 架构特点
编码器-解码器结构

采用标准的Transformer架构
包含编码器（Encoder）和解码器（Decoder）两个主要部分
使用teacher forcing训练机制2
统一的文本到文本范式

输入：文本 + 任务前缀
输出：始终是文本形式
例如：
翻译任务："translate English to German: {text}"
摘要任务："summarize: {text}"
分类任务："classify: {text}"

---

## 3. 预训练方法
T5采用了一种称为"span-corruption"的预训练目标：

随机掩盖原始文本中的片段（spans）
用特殊标记<X>替换这些片段
训练模型预测这些被掩盖的片段
这种方法比BERT的MLM（Masked Language Modeling）更有效

---

## 4. 主要变体
T5-Small: 60M参数
T5-Base: 220M参数
T5-Large: 770M参数
T5-3B: 3B参数
T5-11B: 11B参数

---
## 5. 技术创新
- 统一框架
 - 首次提出将所有NLP任务统一到同一个框架下
 - 简化了模型的使用和部署流程
- 改进的预训练策略

使用C4（Colossal Clean Crawled Corpus）数据集
采用更有效的预训练目标
引入任务前缀来区分不同任务
