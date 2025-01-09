## Introduction
This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

## Original Project
- For the original code of this project, you can refer to the: [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master)
- However, their project requirements are too old (python==3.6, pytorch==1.3, ...), and some of them are not reachable from the main-stream PyPI source mirror (I've tried pypi, aliyun, qinghua, 163, tencent,...you name it).
- Hence, we use modern packages to do a little modifications.
