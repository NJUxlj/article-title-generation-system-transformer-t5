import sys
import torch
import os
import random
import os
import numpy as np
import time
import logging
import json

from config import Config

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




	# 加载模型



	# gpu transfer



	# optimizer


	# training data


	# evaluator


	# loss



	# start training


	



if __name__ == '__main__':
	main(Config)
