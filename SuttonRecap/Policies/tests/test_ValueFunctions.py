from unittest import TestCase
import torch.nn as nn

from Policies.ValueFunctions import NNValueFunction


class FakeNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return 1


def transform_in(state, action):
    return None


class TestNNValueFunction(TestCase):
    fakeNN = FakeNN()
    nn_approx = NNValueFunction(lambda x: [1, 2], transform_in, fakeNN)

    def test_get(self):
        self.assertEqual(1, self.nn_approx.get(1, 1))

    def test_get_all(self):
        self.assertEqual({1: 1, 2: 1}, self.nn_approx.get_all(1))
