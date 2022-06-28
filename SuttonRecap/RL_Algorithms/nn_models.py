from typing import List

import torch
from torch import nn


class LinearNN(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)

    def forward(self, xx):
        xx = self.fc1(xx)
        return xx


class SmallNN(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: tuple = (64, 32)):
        super(SmallNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], out_size)

    def forward(self, xx):
        xx = self.relu(self.fc1(xx))
        xx = self.relu(self.fc2(xx))
        xx = self.fc3(xx)
        return xx


class SmallPolicyNN(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: tuple = (64, 32)):
        super(SmallPolicyNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], out_size)

    def forward(self, xx):
        xx = self.relu(self.fc1(xx))
        xx = self.relu(self.fc2(xx))
        xx = torch.functional.F.softmax(self.fc3(xx), dim=-1)
        return xx


class LinearPolicyNN(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(LinearPolicyNN, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)

    def forward(self, xx):
        xx = self.fc1(xx)
        xx = torch.functional.F.softmax(xx, dim=-1)
        return xx


def init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
