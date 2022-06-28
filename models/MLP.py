import numpy as np
import torch
from torch import nn
from skorch import NeuralNetRegressor

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_list = nn.ModuleList([
        nn.Linear(in_dim, 16*out_dim),nn.ReLU(),
        nn.Linear(16*out_dim, 8*out_dim),nn.ReLU(),
        nn.Linear(8*out_dim, 4*out_dim),nn.ReLU(),
        nn.Linear(4*out_dim, 8*out_dim),nn.ReLU(),
        nn.Linear(8*out_dim, 4*out_dim),nn.ReLU(),
        nn.Linear(4*out_dim, 2*out_dim),nn.ReLU(),
        nn.Linear(2*out_dim, 4*out_dim),nn.ReLU(),
        nn.Linear(4*out_dim, 2*out_dim),nn.ReLU(),
        nn.Linear(2*out_dim, 1*out_dim),nn.Tanh()
        ])

    def forward(self, x):
        for module in self.linear_list:
            x = module(x)
        return x




