from collections import namedtuple
from typing import Callable

import torch.nn
from torch import nn, optim


class ValueFunction:
    def get(self, state, action) -> float:
        pass

    def get_all(self, state) -> dict:
        pass

    def put(self, state, action, new_value) -> None:
        pass


class TabularValueFunction(ValueFunction):
    def __init__(self, get_possible_actions: Callable, state_action_function=None) -> None:
        if state_action_function is None:
            state_action_function = {None: {}}
        self.get_possible_actions = get_possible_actions
        self.state_action_function = state_action_function

    def add_if_not_seen(self, state):
        if state not in self.state_action_function:
            self.state_action_function[state] = {}
            for possible_action in self.get_possible_actions(state):
                self.state_action_function[state][possible_action] = 0

    def get(self, state, action) -> float:
        self.add_if_not_seen(state)
        return self.state_action_function[state][action]

    def put(self, state, action, new_value) -> None:
        self.add_if_not_seen(state)
        self.state_action_function[state][action] = new_value

    def get_all(self, state) -> dict:
        self.add_if_not_seen(state)
        return self.state_action_function[state]


class NNValueFunction(ValueFunction):

    def __init__(self, qnn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer = None,
                 criterion=None,
                 batch_size: int = 1):
        self.device = 'cpu'
        self.qnn = qnn.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(qnn.parameters())
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.SmoothL1Loss().to(self.device)
        else:
            self.criterion = criterion

        self.batch_size = batch_size
        self.batch = []
        self.transition = namedtuple('transition', ('state', 'action', 'new_value'))

    def get(self, state, action):
        return self.get_all(state)[action]

    def get_all(self, state):
        self.qnn.eval()
        with torch.no_grad():
            return self.__transform_out(self.qnn(self.__transform_in(state)))

    def put(self, state, action, new_value) -> None:
        tensor_state = self.__transform_in(state)
        self.batch.append(self.transition(tensor_state, torch.tensor([action]), torch.tensor([new_value])))
        if len(self.batch) == self.batch_size:
            self.__batch_update()
            self.batch = []

    def __transform_in(self, state):
        if type(state) == tuple:
            return torch.as_tensor([float(s) for s in state]).to(self.device)
        elif type(state) == list:
            return torch.as_tensor([float(s) for s in state]).to(self.device)
        elif type(state) in (float, int):
            return torch.as_tensor([float(state)]).to(self.device)

    @staticmethod
    def __transform_out(x: torch.Tensor):
        out_dim = x.data.shape[0]
        return {action: float(x[action]) for action in range(out_dim)}

    def __batch_update(self):
        self.qnn.train()

        batch = self.transition(*zip(*self.batch))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        next_value_batch = torch.stack(batch.new_value).to(self.device)

        current_values = self.qnn(state_batch).gather(1, action_batch)
        loss = self.criterion(current_values, next_value_batch)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.qnn.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
