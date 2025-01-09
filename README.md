# Article-Title Generation System based on T5, and traditional Transformer
Write a complete transformer model line by line, and use it to generate the passage title (Seq2Seq)

- we utilize a seq2seq model based on the Transformer in the `transformer` folder, where this model is exactly the same as the one brought up in this paper: [Attention is all you need](http://arxiv.org/abs/1706.03762)

- 使用一个 `<Title, Content>` 数据集进行微调
- The final result is, if you give model a piece of content, it will return a well-written and summarized **`topic`** of this content.

---
## Notice:
- This project is still in the progress, which will be finished in a few days.
--- 



## Environment Configuration
```bash
pip install -r requirements
```



## Run
```bash
python main.py
```




## Result
