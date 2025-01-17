import sys
import torch
import os
import random
import os
import numpy as np
import time
import logging
import json

from config.config import Config

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
	logger.info()

	model = Transformer(

	)

	cuda_flag = torch.cuda.is_available()
	if cuda_flag:
		logger.info("gpu可以使用，迁移模型至GPU")
		model = model.cuda()

	# optimizer
	optimizer = choose_optimizer(config, model)

	# training data
	

	# evaluator


	# loss



	# start training


	



if __name__ == '__main__':
	main(Config)
