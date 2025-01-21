# Article-Title Generation System based on T5, and traditional Transformer
Write a complete transformer model line by line, and use it to generate the passage title (Seq2Seq)

- we utilize a seq2seq model based on the Transformer in the `transformer` folder, where this model is exactly the same as the one brought up in this paper: [Attention is all you need](http://arxiv.org/abs/1706.03762)

- 使用一个 `<Title, Content>` 数据集进行微调
- The final result is, if you give model a piece of content, it will return a well-written and summarized **`topic`** of this content.

---
## Notice:
- This project is still in the progress, which will be finished in a few days.
- the tranformer model is finished, and you can run it by `python main.py`
--- 



## Environment Configuration
```bash
pip install -r requirements
```

## Project Structure
- all the transformer code is in the `transformer` folder, which includes:
```Plain Text
Constants.py
Layers.py
Models.py
Modules.py
Optim.py
SubLayers.py
Translator.py
```

- the main training loop is in the `main.py`.
- the training data for the transformer is in `sample_data.json`


- all T5 related code is in the fold `T5`
- For the training data `imdb` of the t5 model, you should pre-download to the `data` folder, and use the `HFDataset` object in the `data_preprocess.py` to handle it.

- the `tokenization` folder will include the BPE tokenizer and Word Piece tokenizer in the near future.


## Run
```bash
python main.py
```


## Training Snapshot 
![Epoch 199](image/image.png)



## Result
- We did not add any evaluation metrics like Perplexity, BLEU, ROUGE for now ...
