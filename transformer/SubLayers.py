''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
        


class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass
