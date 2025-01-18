import sys
import torch
import torch.nn as nn
import os
import random
import os
import numpy as np
import time
import logging
import json

from config.config import Config
from loader import load_data
from evaluation import Evaluator

from typing import List, Dict, Tuple


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



from transformer.Models import Transformer



def choose_optimizer(config, model):
	optimizer = config['optimizer']
	learning_rate = config['learning_rate']

	if optimizer == 'adam':
		return torch.optim.Adam(param = model.parameters(), lr = learning_rate)
	elif optimizer == 'sgd':
		return torch.optim.Adam(param = model.parameters(), lr = learning_rate)





def main(config):
	'''
	
	'''
	#创建保存模型的目录
	if not os.path.isdir(config['model_path']):
		os.mkdir(config['model_path'])

	# 加载模型
	logger.info(json.dumps(config, ensure_ascii=False, indent=2))

	model = Transformer(
		n_src_vocab=config['vocab_size'],
		n_trg_vocab=config['vocab_size'],
		src_pad_idx = 0,
		trg_pad_idx = 0,
		d_word_vec=512, d_model = 512, d_inner=2048,
		n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,

	)

	cuda_flag = torch.cuda.is_available()
	if cuda_flag:
		logger.info("gpu可以使用，迁移模型至GPU")
		model = model.cuda()

	# optimizer
	optimizer = choose_optimizer(config, model)

	# training data
	train_data = load_data()

	# evaluator
	evaluator = Evaluator(config,model,logger)

	# loss
	loss_func = nn.CrossEntropyLoss(ignore_index=0)

	# start training
	for epoch in range(config['epoch']):
		model.train()
		if cuda_flag:
			model = model.cuda()
		logger.info("epoch %d begin" % epoch)
		train_loss = []
		for index, batch_data in enumerate(train_data):
			if cuda_flag:
				batch_data: List[torch.LongTensor]
				batch_data = [d.cuda() for d in batch_data]
			
			# 解包操作
			# batch_data 中的每个元素都是一个包含三个张量的列表，分别对应 input_seq、target_seq 和 gold。
			input_seq, target_seq, gold = batch_data

			input_seq: torch.LongTensor  # 经过 dataloader后， shape = (batch_size, input_length)
			target_seq: torch.LongTensor # shape = (batch_size, target_length)
			gold: torch.LongTensor # shape = (batch_size, target_length)

			pred = model.forward(input_seq, target_seq) # shape = [B*L, n_trg_vocab]
			loss = loss_func(pred,gold.view(-1))

			train_loss.append(loss.item())
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		logger.info(f"epoch average loss = {np.mean(train_loss)}")
		evaluator.eval(epoch)

	model_path = os.path.join(config['model_path'], f"epoch_{epoch}.pth")
	torch.save(model.state_dict(), model_path)
	return






if __name__ == '__main__':
	main(Config)
