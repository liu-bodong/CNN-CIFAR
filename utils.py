import torch
from torch import nn

def init_weights(x):
    nn.init.xavier_uniform_(x.weight)
    