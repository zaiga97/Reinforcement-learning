from unittest import TestCase

import torch.nn as nn
import torch.nn.functional
import torch

from Policies.ValueFunctions import NNValueFunction
from RL_Environment.RLEnvironments import RLFrozenLake


class SmallNN(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(SmallNN, self).__init__()
        self.fc1 = nn.Linear(in_size, 30)
        self.fc2 = nn.Linear(30, out_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0., std=0.1)


small_NN = SmallNN(2, 4)
small_NN.apply(init_weight)

env = RLFrozenLake()


class TestNN(TestCase):
    value_function = NNValueFunction(small_NN, batch_size=3)

    def test1(self):
        print(self.value_function.get([0, 0], 1))
        print(self.value_function.get_all([0, 0]))

    def test2(self):
        print(self.value_function.get_all([0, 0]))
        self.value_function.put([0, 0], 1, 1)
        self.value_function.put([1, 1], 3, 1)
        self.value_function.put([0, 0], 3, 1000)
        print(self.value_function.get_all([0, 0]))
