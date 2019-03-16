import torch
from torch import nn
import numpy as np

class StaticCoefficientModel(nn.Module):
    def __init__(self, num_mods):
        super(StaticCoefficientModel, self).__init__()
        self.num_mods = num_mods
        self.coefs = nn.Linear(num_mods, 1, bias=False)
        self.coefs.weight.data = torch.FloatTensor(np.zeros((1, num_mods)))

    def forward(self, scores):
        #print(scores)
        return self.coefs(scores)
