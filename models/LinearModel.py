import numpy as np
import torch
from torch import nn
from skorch import NeuralNetRegressor
class LinearRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out




