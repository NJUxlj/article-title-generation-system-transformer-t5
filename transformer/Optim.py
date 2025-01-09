'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
	'''A simple wrapper class for learning rate scheduling'''
	'''一个用于学习率调度的包装类'''

	def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
		'''
        :param optimizer 是传入的优化器对象
  		:param lr_mul: 是学习率的乘数，用于调整学习率的初始值。
		:param d_model 是模型的隐藏层维度，通常用于计算学习率的缩放因子。
  		:param n_warmup_steps 是预热步骤的数量，在训练开始时，学习率会逐渐增加到一个较高的值，然后再逐渐减小。
        :param n_steps 是当前的训练步骤数，初始化为 0。

		'''
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        pass

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        '''
        计算学习率的缩放因子。学习率的缩放因子根据当前的训练步骤数和预热步骤数进行计算。

        a) 基础缩放因子：(d_model ** -0.5)

            这部分与模型维度相关
            维度越大，基础学习率越小，这有助于训练稳定性
        
        b) 动态调整因子：min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        
            预热阶段：当n_steps < n_warmup_steps时，第二项n_steps * n_warmup_steps ** (-1.5)起主导作用
            衰减阶段：当n_steps > n_warmup_steps时，第一项n_steps ** (-0.5)起主导作用

        让我们分析这两个项：

        term1 = n_steps ** (-0.5)
        
            这是一个单调递减函数
            随着步数增加而减小
            减小速率遵循平方根反比关系
        term2 = n_steps * n_warmup_steps ** (-1.5)
        
            这是一个线性增长函数（因为n_warmup_steps是常数）
            可以重写为：(n_steps/n_warmup_steps) * (1/sqrt(n_warmup_steps))
            随着步数线性增加
        
        关键转折点分析：
            让我们找到这两个函数的交叉点，即预热阶段和衰减阶段的分界点：n_steps ** (-0.5) = n_steps * n_warmup_steps ** (-1.5)  
            通过数学推导，可以证明这个交叉点恰好在 n_steps = n_warmup_steps 时发生。
        
        因此，学习率的变化可以分为两个阶段：
            预热阶段 (n_steps < n_warmup_steps)：
                此时 term2 < term1，所以min函数选择term2
                学习率 = (d_model ** -0.5) * (n_steps * n_warmup_steps ** (-1.5))
                这是一个线性增长阶段
                学习率从接近0开始，线性增加
                
            衰减阶段 (n_steps > n_warmup_steps)：
                此时 term1 < term2，所以min函数选择term1
                学习率 = (d_model ** -0.5) * (n_steps ** (-0.5))
                这是一个平方根衰减阶段
                学习率按照平方根的反比例关系逐渐减小

            预热阶段的必要性：
                在训练初期，模型参数随机初始化，梯度可能不稳定
                渐进式增加学习率可以防止训练早期的剧烈震荡
                让模型在较小的学习率下先学习一些基本模式2
            衰减阶段的作用：
                随着训练进行，较小的学习率有助于模型精细调整
                平方根衰减提供了一个相对温和的衰减速度
                有助于模型收敛到更好的局部最优解
        '''
        d_model = self.d_model
        n_steps = self.n_steps
        n_warmup_steps = self.n_warmup_steps

        return (d_model ** -0.5) * min(
            term1 = n_steps ** (-0.5), 
            term2 = n_steps * n_warmup_steps ** (-1.5)
            )   

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps+=1






